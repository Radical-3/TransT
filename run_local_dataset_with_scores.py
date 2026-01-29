import argparse
import os
import numpy as np
import cv2 as cv
from pytracking.evaluation.data import get_local_dataset
from pytracking.evaluation.tracker import trackerlist
from tqdm import tqdm
from pytracking.tracker.transt.config import cfg


def calculate_iou(bbox1, bbox2):
    """
    计算两个边界框的IoU
    bbox格式: [x1, y1, w, h]
    """
    x1_1, y1_1, w1, h1 = bbox1
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    
    x1_2, y1_2, w2, h2 = bbox2
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2
    
    # 计算交集
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # 计算并集
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0


def calculate_iou_loss(pred_bbox, gt_bbox):
    """
    计算IoU损失
    pred_bbox: 预测的边界框 [x1, y1, w, h]
    gt_bbox: 真实的边界框 [x1, y1, w, h]
    返回: IoU损失 (直接使用IoU，因为对抗攻击希望IoU越低越好)
    """
    iou = calculate_iou(pred_bbox, gt_bbox)
    return iou  # 直接返回IoU，因为对抗攻击希望IoU越低越好


def calculate_score_loss(score_map, gt_bbox_in_crop, crop_size, original_size):
    """
    计算得分损失
    score_map: 得分图 (H, W) 或扁平化的 (H*W,)
    gt_bbox_in_crop: 真实边界框在裁剪图像中的坐标 [x1, y1, w, h]
    crop_size: 裁剪图像的大小 (H, W)
    original_size: 原始图像的大小 (H, W)
    返回: 得分损失 (前景最大概率 - 背景最大概率)
    """
    # 处理扁平化的score_map
    if score_map.ndim == 1:
        # TransT使用32x32的窗口，所以得分图应该是32x32
        h, w = 32, 32
        score_map = score_map.reshape(h, w)
    else:
        h, w = score_map.shape
    
    # 计算前景区域对应的token范围
    x1, y1, w_bbox, h_bbox = gt_bbox_in_crop
    
    # 扩大前景区域1.5倍
    expand_factor = 1.5
    center_x = x1 + w_bbox / 2
    center_y = y1 + h_bbox / 2
    new_w = w_bbox * expand_factor
    new_h = h_bbox * expand_factor
    x1_expanded = center_x - new_w / 2
    y1_expanded = center_y - new_h / 2
    
    # 将边界框坐标映射到score_map的网格
    # score_map通常是 H/4 x W/4 的大小
    scale_h = h / crop_size[0]
    scale_w = w / crop_size[1]
    
    x1_scaled = int(x1_expanded * scale_w)
    y1_scaled = int(y1_expanded * scale_h)
    x2_scaled = int((x1_expanded + new_w) * scale_w)
    y2_scaled = int((y1_expanded + new_h) * scale_h)
    
    # 确保边界在有效范围内
    x1_scaled = max(0, min(x1_scaled, w - 1))
    y1_scaled = max(0, min(y1_scaled, h - 1))
    x2_scaled = max(0, min(x2_scaled, w))
    y2_scaled = max(0, min(y2_scaled, h))
    
    # 创建前景掩码
    foreground_mask = np.zeros((h, w), dtype=bool)
    foreground_mask[y1_scaled:y2_scaled, x1_scaled:x2_scaled] = True
    
    # 计算前景区域的最大得分
    foreground_scores = score_map[foreground_mask]
    if len(foreground_scores) > 0:
        foreground_max = np.max(foreground_scores)
    else:
        foreground_max = 0.0
    
    # 计算背景区域的最大得分
    background_scores = score_map[~foreground_mask]
    if len(background_scores) > 0:
        background_max = np.max(background_scores)
    else:
        background_max = 0.0
    
    # 得分损失: 前景最大值 + 2 - 背景最大值
    # 这样可以增大损失值，使其在优化过程中更加显著
    score_loss = foreground_max + 2 - background_max
    
    return score_loss


def calculate_attention_loss(attn_weights, gt_bbox_in_crop, crop_size):
    """
    计算注意力损失
    attn_weights: 注意力权重 (num_heads, query_tokens, key_tokens)
    gt_bbox_in_crop: 真实边界框在裁剪图像中的坐标 [x1, y1, w, h]
    crop_size: 裁剪图像的大小 (H, W)
    返回: 注意力损失 (前景token对自己和模板的注意力权重之和)
    """
    num_heads, num_query_tokens, num_key_tokens = attn_weights.shape
    
    # 计算前景区域对应的token范围
    x1, y1, w_bbox, h_bbox = gt_bbox_in_crop
    
    # 假设搜索区域被划分为 H/16 x W/16 的token网格
    token_h = crop_size[0] // 16
    token_w = crop_size[1] // 16
    
    # 将边界框坐标映射到token网格
    x1_token = int(x1 / crop_size[1] * token_w)
    y1_token = int(y1 / crop_size[0] * token_h)
    x2_token = int((x1 + w_bbox) / crop_size[1] * token_w)
    y2_token = int((y1 + h_bbox) / crop_size[0] * token_h)
    
    # 确保边界在有效范围内
    x1_token = max(0, min(x1_token, token_w - 1))
    y1_token = max(0, min(y1_token, token_h - 1))
    x2_token = max(0, min(x2_token, token_w))
    y2_token = max(0, min(y2_token, token_h))
    
    # 创建前景token的索引
    foreground_token_indices = []
    for y in range(y1_token, y2_token):
        for x in range(x1_token, x2_token):
            token_idx = y * token_w + x
            if token_idx < num_query_tokens:
                foreground_token_indices.append(token_idx)
    
    if len(foreground_token_indices) == 0:
        return 0.0
    
    # 计算前景token对自己和模板的注意力权重
    # 假设模板的token在key_tokens的前面部分
    template_tokens = num_key_tokens // 2  # 假设前一半是模板token
    
    total_attention = 0.0
    for head_idx in range(num_heads):
        for token_idx in foreground_token_indices:
            # 对自己的注意力权重
            self_attention = attn_weights[head_idx, token_idx, token_idx]
            # 对模板的注意力权重
            template_attention = np.sum(attn_weights[head_idx, token_idx, :template_tokens])
            total_attention += (self_attention + template_attention)
    
    # 平均每个前景token的注意力权重
    attention_loss = total_attention / (len(foreground_token_indices) * num_heads)
    
    return attention_loss


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
                intermediate_attn_weights = out.get('intermediate_attn_weights', None)
                
                # 获取真实边界框
                gt_bbox = seq.ground_truth_rect[frame_num]
                
                # 计算IoU损失
                iou_loss = calculate_iou_loss(bbox, gt_bbox)
                
                # 计算得分损失和注意力损失
                score_loss = None
                attention_loss = None
                
                if score_map is not None:
                    # 计算真实边界框在裁剪图像中的坐标
                    # 按照TransT的裁剪逻辑计算
                    crop_size = cfg.TRACK.INSTANCE_SIZE  # TransT的INSTANCE_SIZE
                    original_size = frame.shape[:2]
                    
                    # 计算裁剪区域，与TransT的逻辑一致
                    w_x = bbox[2] + (4 - 1) * ((bbox[2] + bbox[3]) * 0.5)
                    h_x = bbox[3] + (4 - 1) * ((bbox[2] + bbox[3]) * 0.5)
                    s_x = int(np.ceil(np.sqrt(w_x * h_x)))
                    
                    # 计算裁剪区域的中心位置
                    center_pos = np.array([bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2])
                    
                    # 计算真实边界框在裁剪区域中的坐标
                    # 按照SiameseTracker.get_subwindow的逻辑
                    c = (s_x + 1) / 2
                    context_xmin = np.floor(center_pos[0] - c + 0.5)
                    context_ymin = np.floor(center_pos[1] - c + 0.5)
                    
                    # 计算真实边界框在裁剪前的原始搜索区域中的坐标
                    gt_bbox_in_original = [
                        gt_bbox[0] - context_xmin,
                        gt_bbox[1] - context_ymin,
                        gt_bbox[2],
                        gt_bbox[3]
                    ]
                    
                    # 将坐标映射到调整大小后的裁剪图像 (crop_size x crop_size)
                    scale = crop_size / s_x
                    gt_bbox_in_crop = [
                        gt_bbox_in_original[0] * scale,
                        gt_bbox_in_original[1] * scale,
                        gt_bbox_in_original[2] * scale,
                        gt_bbox_in_original[3] * scale
                    ]
                    
                    # 确保坐标为正数
                    gt_bbox_in_crop = [max(0, x) for x in gt_bbox_in_crop]
                    
                    # 调试信息：打印前景和背景的最大值
                    h, w = 32, 32
                    reshaped_score_map = score_map.reshape(h, w)
                    
                    # 计算前景区域对应的token范围（使用扩大后的区域）
                    x1, y1, w_bbox, h_bbox = gt_bbox_in_crop
                    
                    # 扩大前景区域1.5倍
                    expand_factor = 1.5
                    center_x = x1 + w_bbox / 2
                    center_y = y1 + h_bbox / 2
                    new_w = w_bbox * expand_factor
                    new_h = h_bbox * expand_factor
                    x1_expanded = center_x - new_w / 2
                    y1_expanded = center_y - new_h / 2
                    
                    scale_h = h / crop_size
                    scale_w = w / crop_size
                    
                    x1_scaled = int(x1_expanded * scale_w)
                    y1_scaled = int(y1_expanded * scale_h)
                    x2_scaled = int((x1_expanded + new_w) * scale_w)
                    y2_scaled = int((y1_expanded + new_h) * scale_h)
                    
                    # 确保边界在有效范围内
                    x1_scaled = max(0, min(x1_scaled, w - 1))
                    y1_scaled = max(0, min(y1_scaled, h - 1))
                    x2_scaled = max(0, min(x2_scaled, w))
                    y2_scaled = max(0, min(y2_scaled, h))
                    
                    # 创建前景掩码
                    foreground_mask = np.zeros((h, w), dtype=bool)
                    foreground_mask[y1_scaled:y2_scaled, x1_scaled:x2_scaled] = True
                    
                    # 计算前景和背景的最大值
                    foreground_scores = reshaped_score_map[foreground_mask]
                    background_scores = reshaped_score_map[~foreground_mask]
                    
                    foreground_max = np.max(foreground_scores) if len(foreground_scores) > 0 else 0.0
                    background_max = np.max(background_scores) if len(background_scores) > 0 else 0.0
                    
                    print(f"Debug - Foreground mask shape: {foreground_mask.shape}")
                    print(f"Debug - Foreground area: {np.sum(foreground_mask)} pixels")
                    print(f"Debug - Background area: {h*w - np.sum(foreground_mask)} pixels")
                    print(f"Debug - Foreground max: {foreground_max:.4f}")
                    print(f"Debug - Background max: {background_max:.4f}")
                    print(f"Debug - Foreground mean: {np.mean(foreground_scores):.4f}")
                    print(f"Debug - Background mean: {np.mean(background_scores):.4f}")
                    
                    # 计算得分损失
                    score_loss = calculate_score_loss(score_map, gt_bbox_in_crop, (crop_size, crop_size), original_size)
                
                if attn_weights is not None:
                    # 计算注意力损失
                    # 使用与得分损失相同的裁剪逻辑
                    crop_size = cfg.TRACK.INSTANCE_SIZE  # TransT的INSTANCE_SIZE
                    
                    # 计算裁剪区域，与TransT的逻辑一致
                    w_x = bbox[2] + (4 - 1) * ((bbox[2] + bbox[3]) * 0.5)
                    h_x = bbox[3] + (4 - 1) * ((bbox[2] + bbox[3]) * 0.5)
                    s_x = int(np.ceil(np.sqrt(w_x * h_x)))
                    
                    # 计算裁剪区域的中心位置
                    center_pos = np.array([bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2])
                    
                    # 计算真实边界框在裁剪区域中的坐标
                    c = (s_x + 1) / 2
                    context_xmin = np.floor(center_pos[0] - c + 0.5)
                    context_ymin = np.floor(center_pos[1] - c + 0.5)
                    
                    # 计算真实边界框在裁剪前的原始搜索区域中的坐标
                    gt_bbox_in_original = [
                        gt_bbox[0] - context_xmin,
                        gt_bbox[1] - context_ymin,
                        gt_bbox[2],
                        gt_bbox[3]
                    ]
                    
                    # 将坐标映射到调整大小后的裁剪图像 (crop_size x crop_size)
                    scale = crop_size / s_x
                    gt_bbox_in_crop = [
                        gt_bbox_in_original[0] * scale,
                        gt_bbox_in_original[1] * scale,
                        gt_bbox_in_original[2] * scale,
                        gt_bbox_in_original[3] * scale
                    ]
                    
                    # 确保坐标为正数
                    gt_bbox_in_crop = [max(0, x) for x in gt_bbox_in_crop]
                    
                    attention_loss = calculate_attention_loss(attn_weights, gt_bbox_in_crop, (crop_size, crop_size))
                
                # 计算中间层注意力损失
                intermediate_attention_loss = None
                if intermediate_attn_weights is not None and len(intermediate_attn_weights) > 0:
                    # 计算裁剪区域，与TransT的逻辑一致
                    crop_size = cfg.TRACK.INSTANCE_SIZE  # TransT的INSTANCE_SIZE
                    
                    # 计算裁剪区域，与TransT的逻辑一致
                    w_x = bbox[2] + (4 - 1) * ((bbox[2] + bbox[3]) * 0.5)
                    h_x = bbox[3] + (4 - 1) * ((bbox[2] + bbox[3]) * 0.5)
                    s_x = int(np.ceil(np.sqrt(w_x * h_x)))
                    
                    # 计算裁剪区域的中心位置
                    center_pos = np.array([bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2])
                    
                    # 计算真实边界框在裁剪区域中的坐标
                    c = (s_x + 1) / 2
                    context_xmin = np.floor(center_pos[0] - c + 0.5)
                    context_ymin = np.floor(center_pos[1] - c + 0.5)
                    
                    # 计算真实边界框在裁剪前的原始搜索区域中的坐标
                    gt_bbox_in_original = [
                        gt_bbox[0] - context_xmin,
                        gt_bbox[1] - context_ymin,
                        gt_bbox[2],
                        gt_bbox[3]
                    ]
                    
                    # 将坐标映射到调整大小后的裁剪图像 (crop_size x crop_size)
                    scale = crop_size / s_x
                    gt_bbox_in_crop = [
                        gt_bbox_in_original[0] * scale,
                        gt_bbox_in_original[1] * scale,
                        gt_bbox_in_original[2] * scale,
                        gt_bbox_in_original[3] * scale
                    ]
                    
                    # 确保坐标为正数
                    gt_bbox_in_crop = [max(0, x) for x in gt_bbox_in_crop]
                    
                    # 计算所有中间层的注意力损失
                    total_intermediate_loss = 0.0
                    for layer_name, layer_weights in intermediate_attn_weights.items():
                        # 跳过None值
                        if layer_weights is None:
                            continue
                        
                        # 计算当前层的注意力损失
                        layer_loss = calculate_attention_loss(layer_weights, gt_bbox_in_crop, (crop_size, crop_size))
                        total_intermediate_loss += layer_loss
                        print(f"Debug - {layer_name} attention loss: {layer_loss:.4f}")
                    
                    # 平均所有中间层的注意力损失
                    if len(intermediate_attn_weights) > 0:
                        intermediate_attention_loss = total_intermediate_loss / len(intermediate_attn_weights)
                        print(f"Debug - Intermediate attention loss: {intermediate_attention_loss:.4f}")
                
                # 输出跟踪结果
                print(f"\nFrame {frame_num}/{len(seq.frames)-1}:")
                print(f"Predicted bounding box: {bbox}")
                print(f"Ground truth bounding box: {gt_bbox}")
                print(f"IoU loss: {iou_loss:.4f}")
                if best_score is not None:
                    print(f"Best score: {np.max(best_score):.4f}")
                if score_map is not None:
                    print(f"Score map shape: {score_map.shape}")
                if score_loss is not None:
                    print(f"Score loss: {score_loss:.4f}")
                if attention_loss is not None:
                    print(f"Attention loss: {attention_loss:.4f}")
                if intermediate_attention_loss is not None:
                    print(f"Intermediate attention loss: {intermediate_attention_loss:.4f}")
                if attn_weights is not None:
                    print(f"Attention weights shape: {attn_weights.shape}")
                if intermediate_attn_weights is not None and len(intermediate_attn_weights) > 0:
                    print(f"Number of intermediate attention layers: {len(intermediate_attn_weights)}")
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
