import os
import sys
import argparse

# 添加项目根目录到 Python 路径
env_path = os.path.abspath(os.path.dirname(__file__))
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.experiments.evaluate_transt import evaluate_transt_got10k, evaluate_transt_lasot, evaluate_transt_otb
from pytracking.analysis.plot_results import plot_results, print_results


def evaluate_results(dataset_name):
    """Evaluate tracking results and generate analysis"""
    print(f"Evaluating TransT tracker on {dataset_name} dataset...")
    
    # 根据数据集名称选择对应的实验函数
    if dataset_name == 'got10k':
        trackers, dataset = evaluate_transt_got10k()
        report_name = 'transt_got10k'
    elif dataset_name == 'lasot':
        trackers, dataset = evaluate_transt_lasot()
        report_name = 'transt_lasot'
    elif dataset_name == 'otb':
        trackers, dataset = evaluate_transt_otb()
        report_name = 'transt_otb'
    else:
        print(f"Unknown dataset: {dataset_name}")
        return
    
    # 生成评估报告
    print("\nGenerating evaluation report...")
    print_results(trackers, dataset, report_name, plot_types=('success', 'prec', 'norm_prec'))
    
    # 生成评估图表
    print("\nGenerating evaluation plots...")
    plot_results(trackers, dataset, report_name, plot_types=('success', 'prec', 'norm_prec'))
    
    print(f"\nEvaluation completed! Results saved in pytracking/result_plots/{report_name}/")


def main():
    parser = argparse.ArgumentParser(description='Evaluate TransT tracking results')
    parser.add_argument('--dataset', type=str, default='got10k', 
                        choices=['got10k', 'lasot', 'otb'],
                        help='Dataset to evaluate on (default: got10k)')
    
    args = parser.parse_args()
    evaluate_results(args.dataset)


if __name__ == '__main__':
    main()
