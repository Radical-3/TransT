import argparse
from pytracking.evaluation.data import get_local_dataset
from pytracking.evaluation.running import run_dataset
from pytracking.evaluation.tracker import trackerlist


def run_local_dataset(dataset_path, split='test'):
    """
    在本地数据集上运行跟踪器
    args:
        dataset_path: 本地数据集根目录路径
        split: 数据集分割，可选 'train' 或 'test'
    """
    # 获取本地数据集
    dataset = get_local_dataset(dataset_path, split)
    
    # 定义要运行的跟踪器
    # 这里使用默认的transt50模型，你可以根据需要修改
    trackers = trackerlist('transt', 'transt50', None)
    
    # 运行跟踪器
    print(f"Running TransT tracker on local dataset ({split})...")
    run_dataset(dataset, trackers, threads=1)
    
    print(f"Tracking completed. Results saved to pytracking/tracking_results/")


def main():
    """
    主函数，解析命令行参数并执行跟踪
    """
    parser = argparse.ArgumentParser(description='Run TransT tracker on local dataset')
    parser.add_argument('--dataset_path', type=str, default='dataset/local_dataset',
                        help='Path to local dataset root directory')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'],
                        help='Dataset split to run (train or test)')
    args = parser.parse_args()
    
    # 执行跟踪
    run_local_dataset(args.dataset_path, args.split)


if __name__ == '__main__':
    main()
