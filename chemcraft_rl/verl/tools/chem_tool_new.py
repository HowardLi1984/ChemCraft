
import json
import logging
import os
import ast
import threading
from contextlib import ExitStack
from enum import Enum
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

import ray
import ray.actor

from verl.tools.utils.chemagent_utils import perform_single_search_batch
from verl.utils.rollout_trace import rollout_trace_op

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


# Adapted from verl/tools/sandbox_fusion_tools.py
class PoolMode(Enum):
    """Execution pool mode enumeration."""

    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """Ray actor for rate limiting using token bucket algorithm."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0  # For observability
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        """Acquire a token from the bucket."""
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        """Release a token back to the bucket."""
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        """Get current number of acquired tokens."""
        return self.current_count


class SearchExecutionWorker:
    """Worker for executing search operations with optional rate limiting."""

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        """Initialize singleton rate limiter."""
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        """Health check method."""
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        """Execute function with optional rate limiting."""
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return fn(*fn_args, **fn_kwargs)
                except Exception as e:
                    # TODO we should make this available to the tool caller
                    logger.warning(f"Error when executing search: {e}")
        else:
            return fn(*fn_args, **fn_kwargs)


def init_search_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode
):
    """Initialize search execution pool."""
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(SearchExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")


## Start Making Chemical Functions

class MolSimilarity(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.function_name = self.__class__.__name__
        self._instance_dict = {}
        # Worker and rate limiting configuration
        self.num_workers = config.get("num_workers", 10)
        self.rate_limit = config.get("rate_limit", 120)
        self.timeout = config.get("timeout", 30)

        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = init_search_execution_pool(
            num_workers=self.num_workers,
            enable_global_rate_limit=self.enable_global_rate_limit,
            rate_limit=self.rate_limit,
            mode=PoolMode.ThreadMode,
        )

        # Retrieval service configuration
        self.retrieval_service_url = config.get("retrieval_service_url")
        assert self.retrieval_service_url, "Configuration must include 'retrieval_service_url'"
        if self.retrieval_service_url == "": raise ValueError("retrieval_service_url is not set")

        logger.info(f"Initialized SearchTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
        }
        return instance_id, ToolResponse()

    def execute_search(self, instance_id: str, query_list: list, retrieval_service_url: str, timeout: int):
        result_text, metadata = perform_single_search_batch(
            retrieval_service_url=retrieval_service_url,
            query_list=query_list,
            concurrent_semaphore=None,  # Ray handles concurrency control
            timeout=timeout,
        )
        logger.debug(f"Search result for instance {instance_id}: {result_text}")
        return result_text, metadata

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        timeout = self.timeout
        function_name = self.function_name
        if function_name == "FailFunctionName":
            error_msg = f'Tool Name Error. DO NOT have this tool. Try again.'
            logger.error(f"[SearchTool] {error_msg} Received parameters-{parameters}-, type={type(parameters)}, function_name={function_name}")
            return ToolResponse(text=json.dumps({"result": error_msg})), 0.0, {}
        try:
            # param_dict = ast.literal_eval(parameters)
            query_list_from_params = {
                'name': function_name,
                'arguments': parameters, 
            } # query_list_from_params: {'name': 'FunctionalGroups', 'arguments': {'SMILES': 'C1CCNCC1'}}
        except:
            tool_template = "<tool_call>{'name': 'function-name', 'arguments': {'query1':'', 'query2':''}}</tool_call>"
            error_msg = f'Tool Call Error. If I want to call chemical tools, I should put the tool query with the template {tool_template}. Try again.'
            logger.error(f"[SearchTool] {error_msg} Received parameters-{parameters}-, type={type(parameters)}, function_name={function_name}")
            return ToolResponse(text=json.dumps({"result": error_msg})), 0.0, {}

        # Execute search using Ray execution pool
        try:
            result_text, metadata = await self.execution_pool.execute.remote(
                self.execute_search, instance_id, query_list_from_params, self.retrieval_service_url, timeout
            )

            # Store results in instance dictionary
            self._instance_dict[instance_id]["reward"].append(result_text.strip())

            # Convert metadata to metrics
            metrics = {
                "query_count": metadata.get("query_count", 0),
                "status": metadata.get("status", "unknown"),
                "total_results": metadata.get("total_results", 0),
                "api_request_error": metadata.get("api_request_error"),
            }

            return ToolResponse(text=result_text), 0.0, metrics

        except Exception as e:
            error_result = json.dumps({"result": f"Search execution failed: {e}"})
            logger.error(f"[SearchTool] Execution failed: {e}")
            return ToolResponse(text=error_result), 0.0, {"error": str(e)}

class SMILES2Weight(MolSimilarity):
    pass 

class FunctionalGroups(MolSimilarity):
    pass 

class CompareSMILES(MolSimilarity):
    pass 

class CanonicalizeSMILES(MolSimilarity):
    pass 

class CountMolAtoms(MolSimilarity):
    pass 

class ReplaceFunctionalGroup(MolSimilarity):
    pass 

class RemoveFunctionalGroup(MolSimilarity):
    pass 

class AddFunctionalGroup(MolSimilarity):
    pass 

class GetRXN(MolSimilarity):
    pass 

class GetRXNTemplate(MolSimilarity):
    pass 

class QEDPropertyPred(MolSimilarity):
    pass 

class DRD2PropertyPred(MolSimilarity):
    pass 

class JNK3PropertyPred(MolSimilarity):
    pass 

class LogPPropertyPred(MolSimilarity):
    pass 

class GSKPropertyPred(MolSimilarity):
    pass 

class SolubilityPropertyPred(MolSimilarity):
    pass

class FailFunctionName(MolSimilarity):
    pass