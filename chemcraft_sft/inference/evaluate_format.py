
import json
import matplotlib.pyplot as plt
import numpy as np

from main_infer import evaluate_step_structure

def draw_plot(steps, step_valid, final_answer_rate, count_number):
    plt.figure(figsize=(10, 5))  # 设置图形大小

    # 绘制三条曲线
    # plt.plot(steps, step_valid, marker='o', label='Step Valid', color='blue')
    # plt.plot(steps, final_answer_rate, marker='s', label='Final Answer Rate', color='green')
    plt.plot(steps, count_number, marker='^', label='Count Numer', color='red')

    # 添加标题和标签
    plt.title('Performance Metrics by Step', fontsize=14)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Value', fontsize=12)

    # 添加图例
    plt.legend(fontsize=10)

    # 设置横坐标刻度
    plt.xticks(steps)

    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.6)

    # 显示图形
    plt.tight_layout()  # 自动调整子图参数
    plt.savefig("cold_start.jpg", bbox_inches='tight', dpi=300)

def evaluate_format():
    step_list, step_valid_list, final_answer_list, count_num_list = list(), list(), list(), list()
    for i in range(1, 21):
        final_eval = {'is_step_valid': 0, 'is_final_answer': 0, 'step_count': 0}
        result_info = json.load(open(f"../results/sft_format/result_step_{str(i*10)}.json", "r"))
        
        for result_text in result_info:
            eval_output = evaluate_step_structure(result_text)
            if eval_output['is_step_valid']:
                final_eval['is_step_valid'] += 1
            if eval_output['is_final_answer']:
                final_eval['is_final_answer'] += 1
            
            final_eval['step_count'] += eval_output['step_count']
        
        print(f"step={str(i*10)}, step_valid: {final_eval['is_step_valid']/len(result_info)}, final_valid: {final_eval['is_final_answer']/len(result_info)}, step_count: {final_eval['step_count']/len(result_info)}")
        
        step_list.append(int(i*10))
        step_valid_list.append(final_eval['is_step_valid']/len(result_info))
        final_answer_list.append(final_eval['is_final_answer']/len(result_info))
        count_num_list.append(final_eval['step_count']/len(result_info))

    draw_plot(step_list, step_valid_list, final_answer_list, count_num_list)

evaluate_format()