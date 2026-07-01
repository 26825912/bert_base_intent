"""
类别权重计算工具 - 处理数据不均衡问题
"""
import torch
import numpy as np
from collections import Counter
from typing import Dict, List, Union
import logging

logger = logging.getLogger(__name__)


def compute_class_weights(labels: Union[List[int], np.ndarray],
                         method: str = 'effective',
                         beta: float = 0.99,
                         max_weight: float = None) -> torch.Tensor:
    """
    计算类别权重张量
    Args:
        labels: 训练集标签列表
        method: 权重计算方法
            - 'balanced': 逆频率权重（中等不均衡）
            - 'effective': 有效样本数权重（极端不均衡，推荐）
            - 'sqrt': 平方根权重（温和）
            - 'log': 对数权重（最温和）
        beta: 有效样本数方法的beta参数（0.9-0.999）
        max_weight: 最大权重限制，防止极小类权重过大
    Returns:
        权重张量 shape: (num_classes,)
    """
    counter = Counter(labels)
    num_samples = len(labels)
    num_classes = len(counter)

    # 计算权重
    weights = {}

    if method == 'balanced':
        for cls, count in counter.items():
            weights[cls] = num_samples / (num_classes * count)

    elif method == 'effective':
        for cls, count in counter.items():
            weights[cls] = (1.0 - beta) / (1.0 - beta ** count)

    elif method == 'sqrt':
        max_count = max(counter.values())
        for cls, count in counter.items():
            weights[cls] = np.sqrt(max_count / count)

    elif method == 'log':
        max_count = max(counter.values())
        for cls, count in counter.items():
            weights[cls] = np.log1p(max_count / count)

    else:
        raise ValueError(f"未知的权重计算方法: {method}")

    # 应用最大权重限制
    if max_weight is not None:
        weights = {k: min(v, max_weight) for k, v in weights.items()}

    # 归一化
    total_weight = sum(weights.values())
    weights = {k: v / total_weight * num_classes for k, v in weights.items()}

    # 转换为张量
    weight_list = [weights.get(i, 1.0) for i in range(num_classes)]
    weight_tensor = torch.tensor(weight_list, dtype=torch.float32)

    # 打印统计信息
    min_w, max_w = min(weights.values()), max(weights.values())
    imbalance_ratio = max(counter.values()) / min(counter.values())
    logger.info(f"类别权重已计算 [方法:{method}] - 不均衡比:{imbalance_ratio:.1f}:1, 权重比:{max_w/min_w:.1f}:1")

    return weight_tensor
