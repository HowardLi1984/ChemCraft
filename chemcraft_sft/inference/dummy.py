import re
import ast

def check_tool_call_at_end(response_text):
    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = list(re.finditer(pattern, response_text, re.DOTALL))
    if not matches: return None

    last_match = matches[-1]
    if last_match.end() != len(response_text): return None

    dict_string = last_match.group(1).strip()
    string_to_parse = ""
    
    # 这个正则模式会查找 "'name':" 后面跟着可选的空格，然后是一个单引号, 如果这种情况, 修改为'name'形式
    is_quoted_pattern = r"'name':\s*'" 
    if re.search(is_quoted_pattern, dict_string):
        string_to_parse = dict_string
    else:
        string_to_parse = re.sub(r"('name':)\s*(.*?)\s*,", r"\1 '\2',", dict_string)
        
    try:
        extracted_dict = ast.literal_eval(string_to_parse)
        print(f"extracted_dict: ", extracted_dict)
        if isinstance(extracted_dict, dict) and 'name' in extracted_dict.keys() and 'arguments' in extracted_dict.keys():
            return extracted_dict
        else:
            return None # 如果解析出来的不是字典，也返回None
    except (ValueError, SyntaxError):
        return None  # 如果字符串格式错误，无法被解析，则返回None
    
if __name__ == "__main__":
    response_text = "I am Li Hao, <tool_call>\n{'name': 'lihao', 'arguments': 'None'}\n</tool_call>"
    kkk = check_tool_call_at_end(response_text)
    print(kkk)