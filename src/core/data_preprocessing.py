"""
数据预处理脚本 - 将原始JSON数据转换为训练用的token格式
"""
import json
import os
import numpy as np
from transformers import BertTokenizer
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """数据预处理器"""

    def __init__(self, tokenizer_name: str = 'bert-base-chinese', max_length: int = 512):
        """
        初始化预处理器
        Args:
            tokenizer_name: tokenizer名称或路径
            max_length: 最大序列长度
        """
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        logger.info(f"Tokenizer加载完成: {tokenizer_name}")

    def load_json_data(self, file_path: str) -> List[Dict]:
        """
        加载JSON数据
        Args:
            file_path: JSON文件路径
        Returns:
            数据列表
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            # 如果是字典格式，转换为列表
            data = [data]
        elif not isinstance(data, list):
            raise ValueError(f"不支持的数据格式: {type(data)}")

        logger.info(f"加载数据: {file_path}, 样本数: {len(data)}")
        return data

    def create_label_mapping(self, labels: List[str]) -> Tuple[Dict, Dict]:
        """
        创建标签映射
        Args:
            labels: 标签列表
        Returns:
            (label2id, id2label)
        """
        unique_labels = sorted(set(labels))
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        id2label = {idx: label for label, idx in label2id.items()}

        logger.info(f"标签数量: {len(unique_labels)}")
        logger.info(f"标签: {unique_labels}")

        return label2id, id2label

    def tokenize_data(self, texts: List[str], labels: List[int]) -> Dict:
        """
        对文本进行tokenization
        Args:
            texts: 文本列表
            labels: 标签列表
        Returns:
            tokenized数据字典
        """
        logger.info("开始tokenization...")

        # 批量编码
        encodings = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )

        result = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': np.array(labels, dtype=np.int64)
        }

        logger.info(f"Tokenization完成, 样本数: {len(labels)}")
        return result

    def split_data(self, data: List[Dict], train_ratio: float = 0.8,
                   val_ratio: float = 0.1, seed: int = 42) -> Tuple:
        """
        划分训练集、验证集、测试集
        Args:
            data: 数据列表
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            seed: 随机种子
        Returns:
            (train_data, val_data, test_data)
        """
        np.random.seed(seed)
        indices = np.random.permutation(len(data))

        train_end = int(len(data) * train_ratio)
        val_end = train_end + int(len(data) * val_ratio)

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        train_data = [data[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]
        test_data = [data[i] for i in test_indices]

        logger.info(f"数据划分 - 训练集: {len(train_data)}, 验证集: {len(val_data)}, 测试集: {len(test_data)}")

        return train_data, val_data, test_data

    def preprocess_and_save(self, input_file: str, output_dir: str,
                            train_ratio: float = 0.8, val_ratio: float = 0.1,
                            seed: int = 42):
        """
        预处理数据并保存
        Args:
            input_file: 输入JSON文件路径
            output_dir: 输出目录
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            seed: 随机种子
        """
        # 加载数据
        data = self.load_json_data(input_file)

        # 提取文本和标签
        texts = [item['query'] for item in data]
        labels_str = [item['label'] for item in data]

        # 创建标签映射
        label2id, id2label = self.create_label_mapping(labels_str)
        labels = [label2id[label] for label in labels_str]

        # 打印标签分布
        label_counts = Counter(labels_str)
        logger.info("标签分布:")
        for label, count in sorted(label_counts.items()):
            logger.info(f"  {label}: {count}")

        # 划分数据集
        train_data, val_data, test_data = self.split_data(
            list(zip(texts, labels)), train_ratio, val_ratio, seed
        )

        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存标签映射
        label_mapping = {
            'label2id': label2id,
            'id2label': id2label,
            'num_classes': len(label2id)
        }
        mapping_file = output_path / 'label_mapping.json'
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(label_mapping, f, ensure_ascii=False, indent=2)
        logger.info(f"标签映射已保存: {mapping_file}")

        # 处理并保存各个数据集
        for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
            split_texts = [item[0] for item in split_data]
            split_labels = [item[1] for item in split_data]

            # Tokenization
            tokenized = self.tokenize_data(split_texts, split_labels)

            # 保存为numpy格式
            split_dir = output_path / 'tokens' / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            np.save(split_dir / 'input_ids.npy', tokenized['input_ids'])
            np.save(split_dir / 'attention_mask.npy', tokenized['attention_mask'])
            np.save(split_dir / 'labels.npy', tokenized['labels'])

            logger.info(f"{split_name}集已保存: {split_dir}")

        logger.info("=" * 60)
        logger.info("数据预处理完成!")
        logger.info(f"输出目录: {output_dir}")
        logger.info("=" * 60)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='BERT意图分类数据预处理')
    parser.add_argument('--input', type=str, required=True, help='输入JSON文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--tokenizer', type=str, default='bert-base-chinese',
                        help='Tokenizer名称或路径')
    parser.add_argument('--max_length', type=int, default=512, help='最大序列长度')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    args = parser.parse_args()

    preprocessor = DataPreprocessor(
        tokenizer_name=args.tokenizer,
        max_length=args.max_length
    )

    preprocessor.preprocess_and_save(
        input_file=args.input,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
