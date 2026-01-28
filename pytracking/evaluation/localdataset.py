import os
import numpy as np
import pandas as pd
from pytracking.evaluation.data import BaseDataset, SequenceList, Sequence


class MyLocalDataset(BaseDataset):
    """
    自定义本地数据集读取类，适配以下格式：
    数据集根目录/
    ├── train/
    │   ├── 序列文件夹1/
    │   │   ├── 0001.jpg
    │   │   ├── 0002.jpg
    │   │   ├── ...
    │   │   └── groundtruth.txt
    │   ├── 序列文件夹2/
    │   │   ├── 0001.jpg
    │   │   ├── ...
    │   │   └── groundtruth.txt
    │   └── list.txt  # 指定训练序列
    ├── test/
    │   ├── 序列文件夹1/
    │   │   ├── 0001.jpg
    │   │   ├── ...
    │   │   └── groundtruth.txt
    │   └── list.txt  # 指定测试序列
    """
    def __init__(self, dataset_path, split='test'):
        """
        args:
            dataset_path: 你的数据集根目录路径（例如 '/home/user/my_dataset'）
            split: 数据集分割，可选 'train' 或 'test'
        """
        super().__init__()
        self.base_path = dataset_path  # 数据集根目录
        self.split = split  # 选择训练或测试分割
        self.split_path = os.path.join(self.base_path, self.split)  # 分割路径（train或test）
        self.sequence_info_list = self._get_sequence_info_list()  # 获取所有序列信息

    def get_sequence_list(self):
        """返回所有序列的列表"""
        return SequenceList([self._construct_sequence(info) for info in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        """构建单个序列对象，包含帧路径和标注"""
        # 序列文件夹路径
        seq_path = sequence_info['path']
        # 图片文件名格式（数字位数和扩展名）
        nz = sequence_info['nz']  # 数字位数（如4表示0001.jpg）
        ext = sequence_info['ext']  # 扩展名（如jpg）
        # 帧范围
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        # 生成所有帧的路径列表（例如：/root/train/seq1/0001.jpg）
        frames = [
            os.path.join(self.base_path, seq_path, f"{frame_num:0{nz}}.{ext}")
            for frame_num in range(start_frame, end_frame + 1)
        ]

        # 标注文件路径
        anno_path = os.path.join(self.base_path, seq_path, "groundtruth.txt")
        # 读取边界框标注（格式：x, y, w, h，每行对应一帧）
        ground_truth_rect = self._read_groundtruth(anno_path)

        # 返回序列对象
        return Sequence(
            name=sequence_info['name'],
            frames=frames,
            dataset='my_local_dataset',
            ground_truth_rect=ground_truth_rect,
            object_class=sequence_info.get('object_class', 'unknown')  # 可选：目标类别
        )

    def _read_groundtruth(self, anno_path):
        """读取groundtruth.txt中的边界框标注"""
        # 支持逗号分隔的txt文件（x, y, w, h）
        # 参考got10k.py的读取方式，兼容浮点数和整数标注
        gt = pd.read_csv(
            anno_path,
            delimiter=',',
            header=None,
            dtype=np.float32,
            na_filter=False,
            low_memory=False
        ).values
        return gt

    def _get_sequence_info_list(self):
        """遍历数据集分割目录，收集所有序列的信息"""
        sequence_info_list = []
        
        # 读取list.txt文件，获取要使用的序列列表
        list_file = os.path.join(self.split_path, "list.txt")
        selected_sequences = []
        
        if os.path.exists(list_file):
            # 读取list.txt文件，每行一个序列名称
            with open(list_file, 'r', encoding='utf-8') as f:
                selected_sequences = [line.strip() for line in f if line.strip()]
            print(f"从 {list_file} 加载了 {len(selected_sequences)} 个序列")
        else:
            # 如果list.txt不存在，使用目录下所有文件夹作为序列
            print(f"警告：{list_file} 不存在，将使用所有子文件夹作为序列")
            selected_sequences = [d for d in os.listdir(self.split_path) 
                                if os.path.isdir(os.path.join(self.split_path, d))]
        
        # 遍历选中的序列
        for seq_name in selected_sequences:
            seq_path = os.path.join(self.split, seq_name)  # 相对路径，如 train/seq1
            full_seq_path = os.path.join(self.base_path, seq_path)  # 绝对路径
            
            # 只处理文件夹
            if not os.path.isdir(full_seq_path):
                print(f"警告：{seq_name} 不是文件夹，已跳过")
                continue
                
            # 检查标注文件是否存在
            anno_file = os.path.join(full_seq_path, "groundtruth.txt")
            if not os.path.exists(anno_file):
                print(f"警告：序列 {seq_name} 缺少 groundtruth.txt，已跳过")
                continue
                
            # 获取所有图片文件，推断命名格式
            img_files = [f for f in os.listdir(full_seq_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
            if not img_files:
                print(f"警告：序列 {seq_name} 缺少图片文件，已跳过")
                continue
                
            # 提取图片文件名中的数字（假设文件名是纯数字+扩展名，如0001.jpg）
            try:
                # 排序图片文件，确保帧顺序正确
                img_files.sort()
                # 取第一张和最后一张图片的编号
                first_img = img_files[0]
                last_img = img_files[-1]
                # 提取数字部分（例如"0001.jpg" -> 1）
                start_frame = int(os.path.splitext(first_img)[0])
                end_frame = int(os.path.splitext(last_img)[0])
                # 推断数字位数（例如"0001"是4位）
                nz = len(os.path.splitext(first_img)[0])
                # 推断扩展名（例如jpg）
                ext = os.path.splitext(first_img)[1][1:]  # 去掉小数点
            except ValueError:
                print(f"警告：序列 {seq_name} 的图片命名格式不规范（需纯数字+扩展名），已跳过")
                continue
                
            # 收集序列信息（可根据需要添加object_class等）
            sequence_info = {
                "name": seq_name,
                "path": seq_path,  # 序列在数据集根目录下的相对路径，如 train/seq1
                "startFrame": start_frame,
                "endFrame": end_frame,
                "nz": nz,  # 数字位数
                "ext": ext,  # 图片扩展名
                "object_class": "unknown"  # 可选：可手动指定或从文件夹名提取
            }
            sequence_info_list.append(sequence_info)
            
        print(f"成功加载 {len(sequence_info_list)} 个序列")
        return sequence_info_list

    def __len__(self):
        return len(self.sequence_info_list)


# 使用示例
if __name__ == "__main__":
    # 替换为你的数据集根目录
    dataset_root = "dataset/local_dataset"
    
    # 加载测试数据集
    print("加载测试数据集...")
    test_dataset = MyLocalDataset(dataset_path=dataset_root, split='test')
    test_sequences = test_dataset.get_sequence_list()
    
    # 打印测试数据集信息
    if test_sequences:
        print(f"测试数据集包含 {len(test_sequences)} 个序列")
        first_test_seq = test_sequences[0]
        print(f"第一个测试序列名称：{first_test_seq.name}")
        print(f"帧数：{len(first_test_seq.frames)}")
        print(f"第一帧路径：{first_test_seq.frames[0]}")
        print(f"第一帧标注：{first_test_seq.ground_truth_rect[0]}")
    
    # 加载训练数据集
    print("\n加载训练数据集...")
    train_dataset = MyLocalDataset(dataset_path=dataset_root, split='train')
    train_sequences = train_dataset.get_sequence_list()
    
    # 打印训练数据集信息
    if train_sequences:
        print(f"训练数据集包含 {len(train_sequences)} 个序列")
        first_train_seq = train_sequences[0]
        print(f"第一个训练序列名称：{first_train_seq.name}")
        print(f"帧数：{len(first_train_seq.frames)}")
        print(f"第一帧路径：{first_train_seq.frames[0]}")
        print(f"第一帧标注：{first_train_seq.ground_truth_rect[0]}")