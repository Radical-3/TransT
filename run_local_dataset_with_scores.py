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
            # 读取npy文件并提取图像数据，设置allow_pickle=True以支持包含对象的数组
            first_frame_data = np.load(seq.frames[0], allow_pickle=True)
            
            # 处理不同格式的npy文件
            if isinstance(first_frame_data, dict):
                # 情况1: 直接是字典格式，包含'image'键
                first_frame = first_frame_data['image']
                print("Info: Using direct dict format")
            elif isinstance(first_frame_data, np.ndarray):
                if first_frame_data.ndim == 0:
                    # 情况2: 维度为0的数组，转换为对象后是字典
                    try:
                        data_obj = first_frame_data.item()
                        if isinstance(data_obj, dict) and 'image' in data_obj:
                            first_frame = data_obj['image']
                            print("Info: Using 0D array converted to dict")
                        else:
                            print(f"Error: 0D array contains unexpected object type: {type(data_obj)}")
                            raise
                    except Exception as e:
                        print(f"Error: Failed to process 0D array: {e}")
                        print(f"File path: {seq.frames[0]}")
                        raise
                elif first_frame_data.shape == (1,):
                    # 情况3: 形状为(1,)的数组，第一个元素是字典
                    data_obj = first_frame_data[0]
                    if isinstance(data_obj, dict) and 'image' in data_obj:
                        first_frame = data_obj['image']
                        print("Info: Using (1,) array with dict element")
                    else:
                        print(f"Error: (1,) array contains unexpected object type: {type(data_obj)}")
                        raise
                elif first_frame_data.shape == (6,):
                    # 情况4: 形状为(6,)的数组，顺序: identifier, image, image_mask, label, camera_position, relative_remove
                    first_frame = first_frame_data[1]
                    print("Info: Using (6,) array with image at index 1")
                elif first_frame_data.ndim == 1:
                    # 其他一维数组情况
                    try:
                        data_obj = first_frame_data.item()
                        if isinstance(data_obj, dict):
                            first_frame = data_obj['image']
                            print("Info: Converted 1D array to dict and extracted image")
                        else:
                            print(f"Error: 1D array contains unexpected object type: {type(data_obj)}")
                            raise
                    except Exception as e:
                        print(f"Error: Failed to process 1D array: {e}")
                        print(f"File path: {seq.frames[0]}")
                        print(f"Array shape: {first_frame_data.shape}")
                        print(f"Array dtype: {first_frame_data.dtype}")
                        raise
                else:
                    # 假设是直接的图像数据
                    first_frame = first_frame_data
                    print(f"Info: Using {first_frame_data.ndim}D array as image data")
            else:
                # 其他格式，尝试转换为字典
                try:
                    data_obj = first_frame_data.item()
                    if isinstance(data_obj, dict) and 'image' in data_obj:
                        first_frame = data_obj['image']
                        print("Info: Converted scalar to dict and extracted image")
                    else:
                        print(f"Error: Scalar contains unexpected object type: {type(data_obj)}")
                        raise
                except Exception as e:
                    print(f"Error: Failed to extract image from npy file: {e}")
                    print(f"File path: {seq.frames[0]}")
                    print(f"Data type: {type(first_frame_data)}")
                    raise
            
            # 确保图像格式正确 (RGB, 0-255, uint8)
            if first_frame.dtype == np.float32:
                first_frame = (first_frame * 255).astype(np.uint8)
            
            # 检查图像维度
            if first_frame.ndim != 3:
                print(f"Error: Image array has incorrect dimensions: {first_frame.ndim}D, expected 3D (H, W, C)")
                print(f"File path: {seq.frames[0]}")
                print(f"Array shape: {first_frame.shape}")
                raise ValueError(f"Expected 3D image array, got {first_frame.ndim}D array")
            
            # 检查通道数
            if first_frame.shape[2] != 3:
                print(f"Warning: Image array has {first_frame.shape[2]} channels, expected 3 (RGB)")
            init_info = seq.init_info()
            out = tracker_instance.initialize(first_frame, init_info)
            
            # 打印初始化信息
            print(f"Initial bounding box: {init_info['init_bbox']}")
            
            # 遍历后续帧
            print("Tracking frames...")
            for frame_num, frame_path in enumerate(tqdm(seq.frames[1:], desc="Processing frames"), start=1):
                # 加载当前帧，设置allow_pickle=True以支持包含对象的数组
                frame_data = np.load(frame_path, allow_pickle=True)
                
                # 处理不同格式的npy文件
                if isinstance(frame_data, dict):
                    # 情况1: 直接是字典格式，包含'image'键
                    frame = frame_data['image']
                elif isinstance(frame_data, np.ndarray):
                    if frame_data.ndim == 0:
                        # 情况2: 维度为0的数组，转换为对象后是字典
                        try:
                            data_obj = frame_data.item()
                            if isinstance(data_obj, dict) and 'image' in data_obj:
                                frame = data_obj['image']
                            else:
                                print(f"Error: 0D array contains unexpected object type: {type(data_obj)}")
                                raise
                        except Exception as e:
                            print(f"Error: Failed to process 0D array: {e}")
                            print(f"File path: {frame_path}")
                            raise
                    elif frame_data.shape == (1,):
                        # 情况3: 形状为(1,)的数组，第一个元素是字典
                        data_obj = frame_data[0]
                        if isinstance(data_obj, dict) and 'image' in data_obj:
                            frame = data_obj['image']
                        else:
                            print(f"Error: (1,) array contains unexpected object type: {type(data_obj)}")
                            raise
                    elif frame_data.shape == (6,):
                        # 情况4: 形状为(6,)的数组，顺序: identifier, image, image_mask, label, camera_position, relative_remove
                        frame = frame_data[1]
                    elif frame_data.ndim == 1:
                        # 其他一维数组情况
                        try:
                            data_obj = frame_data.item()
                            if isinstance(data_obj, dict):
                                frame = data_obj['image']
                            else:
                                print(f"Error: 1D array contains unexpected object type: {type(data_obj)}")
                                raise
                        except Exception as e:
                            print(f"Error: Failed to process 1D array: {e}")
                            print(f"File path: {frame_path}")
                            print(f"Array shape: {frame_data.shape}")
                            print(f"Array dtype: {frame_data.dtype}")
                            raise
                    else:
                        # 假设是直接的图像数据
                        frame = frame_data
                else:
                    # 其他格式，尝试转换为字典
                    try:
                        data_obj = frame_data.item()
                        if isinstance(data_obj, dict) and 'image' in data_obj:
                            frame = data_obj['image']
                        else:
                            print(f"Error: Scalar contains unexpected object type: {type(data_obj)}")
                            raise
                    except Exception as e:
                        print(f"Error: Failed to extract image from npy file: {e}")
                        print(f"File path: {frame_path}")
                        print(f"Data type: {type(frame_data)}")
                        raise
                
                # 确保图像格式正确 (RGB, 0-255, uint8)
                if frame.dtype == np.float32:
                    frame = (frame * 255).astype(np.uint8)
                
                # 检查图像维度
                if frame.ndim != 3:
                    print(f"Error: Image array has incorrect dimensions: {frame.ndim}D, expected 3D (H, W, C)")
                    print(f"File path: {frame_path}")
                    print(f"Array shape: {frame.shape}")
                    raise ValueError(f"Expected 3D image array, got {frame.ndim}D array")
                
                # 检查通道数
                if frame.shape[2] != 3:
                    print(f"Warning: Image array has {frame.shape[2]} channels, expected 3 (RGB)")
                
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
