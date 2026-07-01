"""
模型训练脚本 - BERT意图分类模型训练
使用配置文件和命令行参数
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np
import json
import os
import logging
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.model import BertWithCustomHead
from src.core.load_data import DataLoaderManager
from src.core.visualization import TrainingVisualizer
from config import load_config, update_config_from_args

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设备检测（在初始化时就执行）
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"使用设备: {device}")
if device == 'cuda':
    logger.info(f"CUDA 设备: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA 版本: {torch.version.cuda}")
else:
    logger.info("使用 CPU 设备")


class IntentTrainer:
    """意图分类训练器"""

    def __init__(self, config: Dict):
        """
        初始化训练器

        Args:
            config: 配置字典
        """
        self.config = config
        self.device = torch.device(config['device'])
        self.class_weights = None  # 类别权重张量
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

        # 创建输出目录
        os.makedirs(config['output_dir'], exist_ok=True)

    def compute_class_weights(self, train_labels: List[int], num_labels: int):
        """计算类别权重"""
        if not self.config.get('use_class_weights', False):
            logger.info("未启用类别权重")
            return

        from .class_weight_utils import compute_class_weights
        self.class_weights = compute_class_weights(
            labels=train_labels,
            method=self.config.get('weight_method', 'effective'),
            beta=self.config.get('weight_beta', 0.99),
            max_weight=self.config.get('max_weight', None)
        ).to(self.device)

    def setup_model(self, num_labels: int):
        """
        设置模型

        Args:
            num_labels: 标签数量
        """
        logger.info(f"加载模型: {self.config['model_name']}")

        # 加载BERT基础模型
        bert = BertModel.from_pretrained(self.config['model_name'])
        bert.config.output_hidden_states = True

        # 冻结BERT底层（前N层）
        freeze_layers = self.config.get('freeze_layers', 0)
        for name, param in bert.named_parameters():
            if 'encoder.layer.' in name:
                layer_num = int(name.split('encoder.layer.')[1].split('.')[0])
                if layer_num < freeze_layers:
                    param.requires_grad = False

        # 创建完整模型
        hidden_dropout_prob = self.config.get('hidden_dropout_prob', 0.2)
        label_smoothing = self.config.get('label_smoothing', 0.1)
        num_cls_layers = self.config.get('num_cls_layers', 4)
        use_layer_attention = self.config.get('use_layer_attention', True)
        self.model = BertWithCustomHead(bert, num_labels, hidden_dropout_prob, label_smoothing,
                                        num_cls_layers, use_layer_attention, self.class_weights)
        self.model.to(self.device)

        # 加载tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.config['model_name'])

        # 层差异化学习率
        total_layers = bert.config.num_hidden_layers
        mid_point = total_layers // 2
        grouped_params = [
            {
                'params': [p for n, p in self.model.bert.encoder.layer[:mid_point].named_parameters() if p.requires_grad],
                'lr': self.config['learning_rate'] * 0.05
            },
            {
                'params': [p for n, p in self.model.bert.encoder.layer[mid_point:].named_parameters() if p.requires_grad],
                'lr': self.config['learning_rate'] * 0.25
            },
            {
                'params': [p for n, p in self.model.bert.embeddings.named_parameters() if p.requires_grad],
                'lr': self.config['learning_rate'] * 0.05
            },
            {
                'params': self.model.classifier.parameters(),
                'lr': self.config['learning_rate']
            },
            {
                'params': [self.model.layer_query] if hasattr(self.model, 'layer_query') else [self.model.cls_weights],
                'lr': self.config['learning_rate']
            },
        ]

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        logger.info(f"可训练参数数量: {sum(p.numel() for p in trainable_params):,}")
        logger.info(f"总参数数量: {sum(p.numel() for p in self.model.parameters()):,}")

        # 设置优化器
        self.optimizer = optim.AdamW(grouped_params)

        logger.info(f"模型已加载到设备: {self.device}")

    def train_epoch(self, train_loader: DataLoader, epoch_num: int) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        batch_count = len(train_loader)

        logger.info(f"Epoch {epoch_num} - 训练批次: {batch_count}")
        epoch_batch_start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == batch_count:
                batch_accuracy = accuracy_score(all_labels, all_preds)
                avg_loss = total_loss / (batch_idx + 1)
                batch_time = time.time() - epoch_batch_start_time
                batch_time_str = time.strftime("%M:%S", time.gmtime(batch_time))
                print(f"\rEpoch {epoch_num} | Batch: {batch_idx + 1}/{batch_count} | Loss: {avg_loss:.4f} | Acc: {batch_accuracy:.4f} | Time: {batch_time_str}", end='', flush=True)

        accuracy = accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / len(train_loader)
        print()
        return avg_loss, accuracy

    def evaluate(self, val_loader: DataLoader) -> Tuple[float, float, List, List]:
        """评估模型"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                preds = torch.argmax(logits, dim=1)

                total_loss += loss.item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        return total_loss / len(val_loader), accuracy, all_preds, all_labels

    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        """训练模型"""
        total_steps = len(train_loader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=total_steps
        )

        best_val_accuracy = 0
        patience = self.config.get('patience', 3)
        patience_counter = 0

        logger.info("=" * 60)
        logger.info("开始训练")
        logger.info("=" * 60)
        logger.info(f"早停机制: patience = {patience} (连续{patience}轮验证准确率不改善则停止)")
        if self.config.get('use_class_weights', False):
            logger.info(f"✓ 已启用类别权重损失 [方法: {self.config.get('weight_method', 'effective')}]")
        else:
            logger.info(f"使用标准交叉熵损失（未启用类别权重）")

        training_start_time = time.time()
        logger.info(f"训练开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(training_start_time))}")

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            logger.info("-" * 60)
            logger.info(f"Epoch 开始时间: {time.strftime('%H:%M:%S', time.localtime(epoch_start_time))}")

            train_loss, train_accuracy = self.train_epoch(train_loader, epoch_num=epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_accuracy)

            val_loss, val_accuracy, _, _ = self.evaluate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)

            logger.info(f"训练损失: {train_loss:.4f} | 训练准确率: {train_accuracy:.4f}")
            logger.info(f"验证损失: {val_loss:.4f} | 验证准确率: {val_accuracy:.4f}")

            self.scheduler.step()

            epoch_end_time = time.time()
            epoch_elapsed = epoch_end_time - epoch_start_time
            epoch_time_str = time.strftime("%H:%M:%S", time.gmtime(epoch_elapsed))
            logger.info(f"Epoch 耗时: {epoch_time_str}")

            if val_accuracy >= best_val_accuracy:
                old_best = best_val_accuracy
                best_val_accuracy = val_accuracy
                patience_counter = 0
                self.save_model(os.path.join(self.config['output_dir'], 'best_model'))
                logger.info(f"✓ 保存最佳模型 (验证准确率: {val_accuracy:.4f} >= 历史最佳: {old_best:.4f})")
                logger.info(f"  Patience 计数器重置为 0")
            else:
                patience_counter += 1
                logger.info(f"  Patience 计数器: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    logger.info(f"\n✗ 早停触发：验证准确率连续{patience}轮未改善")
                    logger.info(f"  最终最佳验证准确率: {best_val_accuracy:.4f}")
                    logger.info(f"  早停时的验证准确率: {val_accuracy:.4f}")
                    logger.info(f"  提前结束训练，避免过拟合")
                    break

        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_training_time))
        logger.info(f"\n总训练时间: {total_time_str}")

        history_path = os.path.join(self.config['output_dir'], 'training_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        logger.info(f"训练历史已保存到: {history_path}")

        best_model_path = os.path.join(self.config['output_dir'], 'best_model', 'model_state_dict.bin')
        if not os.path.exists(best_model_path):
            logger.warning(f"警告: 最佳模型文件不存在: {best_model_path}")
            logger.warning("这可能是因为验证准确率从未提升，正在保存最终模型...")
            self.save_model(os.path.join(self.config['output_dir'], 'best_model'))

        return best_val_accuracy

    def load_model(self, load_path: str, num_labels: int):
        """加载模型"""
        logger.info(f"从 {load_path} 加载模型...")

        bert = BertModel.from_pretrained(self.config['model_name'])

        hidden_dropout_prob = self.config.get('hidden_dropout_prob', 0.1)
        num_cls_layers = self.config.get('num_cls_layers', 4)
        use_layer_attention = self.config.get('use_layer_attention', True)
        self.model = BertWithCustomHead(
            bert,
            num_labels,
            hidden_dropout_prob,
            num_cls_layers=num_cls_layers,
            use_layer_attention=use_layer_attention,
            class_weights=self.class_weights
        )

        state_dict = torch.load(os.path.join(load_path, 'model_state_dict.bin'), map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

        logger.info(f"模型已加载到设备: {self.device}")

    def save_model(self, save_path: str):
        """保存模型"""
        try:
            os.makedirs(save_path, exist_ok=True)
            model_file = os.path.join(save_path, 'model_state_dict.bin')
            torch.save(self.model.state_dict(), model_file)
            self.tokenizer.save_pretrained(save_path)

            if not os.path.exists(model_file):
                raise RuntimeError(f"模型文件保存后不存在: {model_file}")

            logger.info(f"模型已保存到: {save_path}")
        except Exception as e:
            logger.error(f"模型保存失败: {e}")
            raise


def set_seed(seed: int = 42):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """主函数 - 训练模型"""
    logger.info("=" * 60)
    logger.info("BERT意图分类模型训练")
    logger.info("=" * 60)

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='BERT意图分类模型训练')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--data_dir', type=str, help='数据目录路径')
    parser.add_argument('--output_dir', type=str, help='模型输出目录')
    parser.add_argument('--model_path', type=str, help='BERT基础模型路径')
    parser.add_argument('--batch_size', type=int, help='批次大小')
    parser.add_argument('--learning_rate', type=float, help='学习率')
    parser.add_argument('--num_epochs', type=int, help='训练轮数')
    args = parser.parse_args()

    # 加载配置
    yaml_config = load_config(args.config)
    config = update_config_from_args(yaml_config, args)

    # 构建训练配置
    train_config = {
        'model_name': config['paths'].get('base_model_path', config['model']['base_model_name']),
        'data_dir': config['paths']['data_dir'],
        'output_dir': config['paths']['output_dir'],
        'batch_size': config['training']['batch_size'],
        'num_epochs': config['training']['num_epochs'],
        'learning_rate': config['training']['learning_rate'],
        'warmup_steps': config['training']['warmup_steps'],
        'max_length': config['training']['max_length'],
        'device': device,
        'patience': config['training']['patience'],
        'seed': config['training']['seed'],
        'freeze_layers': config['training']['freeze_layers'],
        'hidden_dropout_prob': config['model']['hidden_dropout_prob'],
        'label_smoothing': config['model']['label_smoothing'],
        'num_cls_layers': config['model']['num_cls_layers'],
        'use_layer_attention': config['model']['use_layer_attention'],

        # 权重损失函数配置（可选）
        'use_class_weights': config['training'].get('use_class_weights', False),
        'weight_method': config['training'].get('weight_method', 'effective'),
        'weight_beta': config['training'].get('weight_beta', 0.99),
        'max_weight': config['training'].get('max_weight', None),
    }

    logger.info("\n训练配置:")
    for key, value in train_config.items():
        logger.info(f"  {key}: {value}")

    # 设置随机种子
    set_seed(train_config['seed'])

    # 加载训练数据
    train_loader, val_loader, test_loader, label_mapping = \
        DataLoaderManager(train_config).load_data()

    num_labels = label_mapping['num_classes']

    # 训练模型
    trainer = IntentTrainer(train_config)

    # 计算类别权重（如果启用）
    if train_config.get('use_class_weights', False):
        train_labels = []
        for batch in train_loader:
            train_labels.extend(batch['labels'].numpy().tolist())
        trainer.compute_class_weights(train_labels, num_labels)

    trainer.setup_model(num_labels)
    best_val_accuracy = trainer.train(train_loader, val_loader, train_config['num_epochs'])

    # 评估最佳模型
    logger.info("\n" + "=" * 60)
    logger.info("评估最佳模型")
    logger.info("=" * 60)

    trainer.load_model(os.path.join(train_config['output_dir'], 'best_model'), num_labels)

    test_loss, test_accuracy, test_preds, test_labels = trainer.evaluate(test_loader)
    logger.info(f"\n测试损失: {test_loss:.4f}")
    logger.info(f"测试准确率: {test_accuracy:.4f}")

    # 生成可视化
    logger.info("\n生成模型评估可视化...")
    visualizer = TrainingVisualizer(os.path.join(train_config['output_dir'], 'training_history.json'))

    visualizer.print_classification_report(test_labels, test_preds, list(label_mapping['id2label'].values()))

    cm_path = os.path.join(train_config['output_dir'], 'confusion_matrix.png')
    visualizer.plot_confusion_matrix(test_labels, test_preds, list(label_mapping['id2label'].values()), save_path=cm_path)

    logger.info("生成训练曲线可视化...")
    curve_path = os.path.join(train_config['output_dir'], 'training_curve.png')
    visualizer.plot_combined(save_path=curve_path)

    logger.info("\n" + "=" * 60)
    logger.info("训练完成")
    logger.info("=" * 60)
    logger.info(f"最佳验证准确率: {best_val_accuracy:.4f}")
    logger.info(f"测试准确率: {test_accuracy:.4f}")
    logger.info(f"模型保存在: {train_config['output_dir']}/best_model")


if __name__ == "__main__":
    main()
