import argparse
from pytracking.evaluation.data import get_local_dataset
from pytracking.analysis.plot_results import print_results, plot_results
from pytracking.evaluation.tracker import trackerlist


def evaluate_local_dataset(dataset_path, split='test'):
    """
    评估本地数据集上的跟踪结果
    args:
        dataset_path: 本地数据集根目录路径
        split: 数据集分割，可选 'train' 或 'test'
    """
    # 获取本地数据集
    dataset = get_local_dataset(dataset_path, split)
    
    # 定义要评估的跟踪器
    # 这里使用默认的transt50模型，你可以根据需要修改
    trackers = trackerlist('transt', 'transt50', None)
    
    # 生成报告名称
    report_name = f'local_dataset_{split}'
    
    # 打印评估结果
    print(f"Evaluating TransT tracker on local dataset ({split})...")
    print_results(trackers, dataset, report_name, plot_types=('success', 'prec', 'norm_prec'))
    
    # 生成评估图表
    plot_results(trackers, dataset, report_name, plot_types=('success', 'prec', 'norm_prec'))
    
    print(f"Evaluation completed. Results saved to pytracking/results/{report_name}")


def main():
    """
    主函数，解析命令行参数并执行评估
    """
    parser = argparse.ArgumentParser(description='Evaluate TransT tracker on local dataset')
    parser.add_argument('--dataset_path', type=str, default='dataset/local_dataset',
                        help='Path to local dataset root directory')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'],
                        help='Dataset split to evaluate (train or test)')
    args = parser.parse_args()
    
    # 执行评估
    evaluate_local_dataset(args.dataset_path, args.split)


if __name__ == '__main__':
    main()
