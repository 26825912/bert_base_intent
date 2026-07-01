"""
数据加载脚本 - 训练数据加载器
只负责加载data_preprocessing.py预生成的token文件
"""
import json
import os
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# 配置日志（只输出到控制台）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TokenDataset(Dataset):
    """Token数据集"""

    def __init__(self, input_ids: np.ndarray, attention_mask: np.ndarray, labels: np.ndarray):
        """
        Args:
            input_ids: input_ids数组
            attention_mask: attention_mask数组
            labels: 标签数组
        """
        self.input_ids = torch.from_numpy(input_ids)
        self.attention_mask = torch.from_numpy(attention_mask)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }


class DataLoaderManager:
    """数据加载管理器"""

    def __init__(self, config: dict):
        self.config = config
        self.data_dir = config['data_dir']
        self.batch_size = config['batch_size']

    def _load_dataset(self, split: str) -> Dataset:
        """
        加载预生成的token数据集

        Args:
            split: 数据集名称 ('train', 'val', 'test')

        Returns:
            Dataset对象
        """
        token_dir = os.path.join(self.data_dir, 'tokens', split)
        input_ids_path = os.path.join(token_dir, 'input_ids.npy')

        if not os.path.exists(input_ids_path):
            raise FileNotFoundError(f"未找到预生成token文件: {input_ids_path}")

        input_ids = np.load(input_ids_path)
        attention_mask = np.load(os.path.join(token_dir, 'attention_mask.npy'))
        labels = np.load(os.path.join(token_dir, 'labels.npy'))

        logger.info(f"加载{split}集: {len(labels)} 条")

        return TokenDataset(input_ids, attention_mask, labels)

    def load_data(self) -> tuple:
        """
        加载训练数据

        Returns:
            (train_loader, val_loader, test_loader, label_mapping)
        """
        logger.info("=" * 60)
        logger.info("开始加载训练数据")
        logger.info("=" * 60)

        # 加载标签映射
        with open(os.path.join(self.data_dir, 'label_mapping.json'), 'r', encoding='utf-8') as f:
            label_mapping = json.load(f)
        logger.info(f"标签映射已加载，类别数: {label_mapping.get('num_classes', 0)}")

        # 加载数据集
        train_dataset = self._load_dataset('train')
        val_dataset = self._load_dataset('val')
        test_dataset = self._load_dataset('test')

        # 创建DataLoader
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        logger.info(f"数据加载器已创建 (batch_size={self.batch_size})")

        logger.info("=" * 60)
        logger.info("训练数据加载完成")
        logger.info("=" * 60)

        return train_loader, val_loader, test_loader, label_mapping


def main():
    """主函数 - 测试数据加载"""
    config = {
        'data_dir': './data',
        'batch_size': 16,
        'max_length': 512,
        'model_name': 'bert-base-chinese'
    }

    manager = DataLoaderManager(config)
    train_loader, val_loader, test_loader, label_mapping = manager.load_data()

    print("训练数据加载器:", train_loader)
    print("验证数据加载器:", val_loader)
    print("测试数据加载器:", test_loader)
    print("标签映射:", label_mapping)


if __name__ == "__main__":
    main()
