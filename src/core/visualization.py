"""
训练可视化模块
训练过程可视化
"""
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import os
import logging
from matplotlib import font_manager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TrainingVisualizer:
    """训练可视化器"""

    def __init__(self, history_path=None):
        """
        初始化可视化器

        Args:
            history_path: 训练历史文件路径
        """
        self.history = None
        if history_path:
            self.load_history(history_path)

    def load_history(self, history_path):
        """
        加载训练历史

        Args:
            history_path: 训练历史文件路径
        """
        with open(history_path, 'r', encoding='utf-8') as f:
            self.history = json.load(f)
        logger.info(f"从 {history_path} 加载训练历史")

    def plot_training_loss(self, save_path=None):
        """
        绘制训练和验证损失曲线

        Args:
            save_path: 保存路径（可选）
        """
        if not self.history:
            raise ValueError("请先加载训练历史")

        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.history['train_loss']) + 1)

        plt.plot(epochs, self.history['train_loss'], 'b-o', label='训练损失')
        plt.plot(epochs, self.history['val_loss'], 'r-s', label='验证损失')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('训练和验证损失曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"损失曲线已保存到 {save_path}")

        plt.show()

    def plot_training_accuracy(self, save_path=None):
        """
        绘制验证准确率曲线
        Args:
            save_path: 保存路径（可选）
        """
        if not self.history:
            raise ValueError("请先加载训练历史")

        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.history['val_accuracy']) + 1)

        plt.plot(epochs, self.history['val_accuracy'], 'g-o', label='验证准确率')
        plt.axhline(y=max(self.history['val_accuracy']), color='r',
                    linestyle='--', label=f'最佳准确率: {max(self.history["val_accuracy"]):.4f}')

        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('验证准确率曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"准确率曲线已保存到 {save_path}")

        plt.show()

    def plot_combined(self, save_path=None):
        """
        绘制组合图表（损失和准确率）

        Args:
            save_path: 保存路径（可选）
        """
        if not self.history:
            raise ValueError("请先加载训练历史")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        epochs = range(1, len(self.history['train_loss']) + 1)

        # 损失曲线
        ax1.plot(epochs, self.history['train_loss'], 'b-o', label='训练损失')
        ax1.plot(epochs, self.history['val_loss'], 'r-s', label='验证损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('训练和验证损失')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 准确率曲线
        ax2.plot(epochs, self.history['train_accuracy'], 'b-o', label='训练准确率')
        ax2.plot(epochs, self.history['val_accuracy'], 'g-o', label='验证准确率')
        ax2.axhline(y=max(self.history['val_accuracy']), color='r',
                    linestyle='--', label=f'最佳验证: {max(self.history["val_accuracy"]):.4f}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('训练和验证准确率')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"组合图表已保存到 {save_path}")

        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, labels, save_path=None):
        """
        绘制混淆矩阵

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            labels: 标签列表
            save_path: 保存路径（可选）
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"混淆矩阵已保存到 {save_path}")

        plt.show()

    def plot_classification_report(self, y_true, y_pred, labels, save_path=None):
        """
        绘制分类报告

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            labels: 标签列表
            save_path: 保存路径（可选）
        """
        report = classification_report(y_true, y_pred, labels=range(len(labels)),
                                      target_names=labels, output_dict=True)

        # 提取各项指标
        metrics = ['precision', 'recall', 'f1-score']
        data = []
        for label in labels:
            row = []
            for metric in metrics:
                row.append(report[label][metric])
            data.append(row)

        # 绘制柱状图
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(labels))
        width = 0.25

        for i, metric in enumerate(metrics):
            values = [row[i] for row in data]
            offset = (i - 1) * width
            ax.bar(x + offset, values, width, label=metric)

        ax.set_xlabel('意图类别')
        ax.set_ylabel('分数')
        ax.set_title('各意图类别的分类性能')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"分类报告图表已保存到 {save_path}")

        plt.show()

    def print_classification_report(self, y_true, y_pred, labels):
        """
        打印分类报告

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            labels: 标签列表
        """
        report = classification_report(y_true, y_pred, labels=range(len(labels)),
                                      target_names=labels, digits=4)
        logger.info("\n分类报告:")
        logger.info("=" * 80)
        logger.info(report)
        logger.info("=" * 80)


def main():
    """主函数 - 演示可视化功能"""
    logger.info("=" * 50)
    logger.info("训练可视化")
    logger.info("=" * 50)

    # 加载训练历史
    history_path = './models/training_history.json'
    if os.path.exists(history_path):
        visualizer = TrainingVisualizer(history_path)

        # 绘制各种图表
        output_dir = './visualizations'
        os.makedirs(output_dir, exist_ok=True)

        logger.info("\n绘制训练和验证损失曲线...")
        visualizer.plot_training_loss(f'{output_dir}/loss_curve.png')

        logger.info("\n绘制验证准确率曲线...")
        visualizer.plot_training_accuracy(f'{output_dir}/accuracy_curve.png')

        logger.info("\n绘制组合图表...")
        visualizer.plot_combined(f'{output_dir}/combined_chart.png')

    else:
        logger.info(f"训练历史文件 {history_path} 不存在，请先训练模型")


if __name__ == "__main__":
    main()
