import argparse
import os
import numpy as np
import cv2 as cv
from pytracking.evaluation.data import get_local_dataset
from pytracking.evaluation.tracker import trackerlist
from tqdm import tqdm


def run_local_dataset_with实时输出(dataset_path, split='test'):
    """
    在本地数据集上运行跟踪器并实时输出结果
    args:
        dataset_path: 本地数据集根目录路径
        split: 数据集分割，可选 'train' 或 'test'
    """
    # 获取本地数据集
    dataset = get_local_dataset(dataset_path, split)
    
    # 定义要运行的跟踪器
    trackers = trackerlist('transt', 'transt50', None)
    
    # 遍历每个跟踪器
    for tracker in trackers:
        print(f"Running tracker: {tracker.name} {tracker.parameter_name}")
        
        # 遍历每个序列
        for seq in dataset:
            print(f"\nProcessing sequence: {seq.name}")
            
            # 获取序列信息
            print(f"Sequence length: {len(seq.frames)} frames")
            
            # 创建跟踪器实例
            params = tracker.get_parameters()
            tracker_instance = tracker.create_tracker(params)
            
            # 初始化跟踪器
            print("Initializing tracker...")
            first_frame = cv.imread(seq.frames[0])
            first_frame = cv.cvtColor(first_frame, cv.COLOR_BGR2RGB)
            init_info = seq.init_info()
            out = tracker_instance.initialize(first_frame, init_info)
            
            # 打印初始化信息
            print(f"Initial bounding box: {init_info['init_bbox']}")
            
            # 遍历后续帧
            print("Tracking frames...")
            for frame_num, frame_path in enumerate(tqdm(seq.frames[1:], desc="Processing frames"), start=1):
                # 加载当前帧
                frame = cv.imread(frame_path)
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                
                # 运行跟踪器
                info = seq.frame_info(frame_num)
                info['previous_output'] = out
                
                # 运行跟踪
                out = tracker_instance.track(frame, info)
                
                # 提取跟踪结果
                bbox = out['target_bbox']
                score_map = out.get('score_map', None)
                best_score = out.get('best_score', None)
                attn_weights = out.get('attn_weights', None)
                
                # 输出跟踪结果
                print(f"\nFrame {frame_num}/{len(seq.frames)-1}:")
                print(f"Bounding box: {bbox}")
                if best_score is not None:
                    print(f"Best score: {np.max(best_score):.4f}")
                if score_map is not None:
                    print(f"Score map shape: {score_map.shape}")
                    print(f"Score map min: {np.min(score_map):.4f}, max: {np.max(score_map):.4f}, mean: {np.mean(score_map):.4f}")
                if attn_weights is not None:
                    print(f"Attention weights shape: {attn_weights.shape}")
                    print(f"Attention weights min: {np.min(attn_weights):.4f}, max: {np.max(attn_weights):.4f}, mean: {np.mean(attn_weights):.4f}")
                
                # 这里可以添加你的对抗攻击代码
                # 使用 bbox 和 score_map 构造损失函数
                
            print(f"\nTracking completed for sequence: {seq.name}")


def main():
    """
    主函数，解析命令行参数并执行跟踪
    """
    # parser = argparse.ArgumentParser(description='Run TransT tracker on local dataset with real-time output')
    # parser.add_argument('--dataset_path', type=str, default='dataset/local_dataset',
    #                     help='Path to local dataset root directory')
    # parser.add_argument('--split', type=str, default='test', choices=['train', 'test'],
    #                     help='Dataset split to run (train or test)')
    # args = parser.parse_args()
    
    # 执行跟踪
    # run_local_dataset_with实时输出(args.dataset_path, args.split)
    run_local_dataset_with实时输出('dataset/local_dataset', 'test')


if __name__ == '__main__':
    main()
