# 跟踪结果评测指南

本指南介绍如何评测 `pytracking/tracking_results` 目录下的 TransT 跟踪器结果。

## 目录结构

- `pytracking/tracking_results/` - 存储跟踪结果
- `pytracking/result_plots/` - 存储评测结果和图表
- `evaluate_tracking_results.py` - 评测脚本
- `pytracking/experiments/evaluate_transt.py` - 评测实验定义

## 运行评测

### 步骤 1: 确保跟踪结果存在

确保 `pytracking/tracking_results/transt/transt50/` 目录下存在跟踪结果文件，文件格式为 `{sequence_name}.txt`，每行包含边界框坐标 `x, y, width, height`。

### 步骤 2: 运行评测脚本

在项目根目录运行以下命令：

```bash
# 评测 GOT-10k 验证集
python evaluate_tracking_results.py --dataset got10k

# 评测 LaSOT 数据集
python evaluate_tracking_results.py --dataset lasot

# 评测 OTB 数据集
python evaluate_tracking_results.py --dataset otb
```

### 步骤 3: 查看评测结果

评测完成后，结果会存储在 `pytracking/result_plots/` 目录下，每个数据集对应一个子目录：

- `pytracking/result_plots/transt_got10k/` - GOT-10k 评测结果
- `pytracking/result_plots/transt_lasot/` - LaSOT 评测结果
- `pytracking/result_plots/transt_otb/` - OTB 评测结果

每个目录包含以下文件：

- `eval_data.pkl` - 评测数据
- `success_plot.pdf` - 成功曲线图
- `precision_plot.pdf` - 精度曲线图
- `norm_precision_plot.pdf` - 归一化精度曲线图

## 评测指标

评测脚本会生成以下指标：

1. **AUC** - 成功曲线下的面积，综合评估指标
2. **OP50** - 重叠阈值为 0.5 时的成功率
3. **OP75** - 重叠阈值为 0.75 时的成功率
4. **Precision** - 中心位置误差小于 20 像素的帧数比例
5. **Norm Precision** - 归一化中心位置误差小于 0.5 的帧数比例

## 示例输出

运行评测脚本后，控制台会输出类似以下的评测报告：

```
Evaluating TransT tracker on got10k dataset...

Reporting results over 180 / 180 sequences

                                  | AUC        | OP50       | OP75       | Precision  | Norm Precision |
transt_transt50 [63.5]            | 63.50      | 83.70      | 48.90      | 78.20      | 86.50          |
```

## 自定义评测

如果需要评测其他跟踪器或数据集，可以修改 `pytracking/experiments/evaluate_transt.py` 文件，添加新的实验函数。

例如，要评测其他跟踪器，可以使用 `trackerlist` 函数创建不同的跟踪器实例：

```python
def evaluate_other_tracker():
    """Evaluate other tracker"""
    trackers = trackerlist('tracker_name', 'parameter_name', range(1))
    dataset = get_dataset('dataset_name')
    return trackers, dataset
```

然后在 `evaluate_tracking_results.py` 中添加对应的处理逻辑。
