"""
BERT模型定义 - 自定义分类头
"""
import torch
import torch.nn as nn
from transformers import BertModel


class BertWithCustomHead(nn.Module):
    """BERT + 自定义分类头 (多层CLS融合)"""

    def __init__(self, bert_model, num_labels, hidden_dropout_prob=0.2, label_smoothing=0.1,
                 num_cls_layers=4, use_layer_attention=True, class_weights=None):
        """
        初始化模型

        Args:
            bert_model: BERT基础模型
            num_labels: 分类标签数量
            hidden_dropout_prob: Dropout概率
            label_smoothing: 标签平滑系数
            num_cls_layers: 用于融合的CLS层数
            use_layer_attention: 是否使用注意力机制融合
            class_weights: 类别权重张量（用于处理数据不均衡）
        """
        super().__init__()
        self.bert = bert_model
        self.label_smoothing = label_smoothing
        self.num_cls_layers = num_cls_layers
        self.use_layer_attention = use_layer_attention
        self.class_weights = class_weights

        bert_hidden_size = bert_model.config.hidden_size

        # CLS层融合方式
        if use_layer_attention:
            # 使用可学习的query向量进行注意力融合
            self.layer_query = nn.Parameter(torch.randn(1, 1, bert_hidden_size) * 0.02)
        else:
            # 使用可学习的权重进行加权融合
            self.cls_weights = nn.Parameter(torch.ones(num_cls_layers) / num_cls_layers)

        self.dropout = nn.Dropout(hidden_dropout_prob)

        # 分类头: 768 → 384 → 192 → num_labels
        self.classifier = nn.Sequential(
            nn.Linear(bert_hidden_size, bert_hidden_size // 2),
            nn.LayerNorm(bert_hidden_size // 2),
            nn.GELU(),
            nn.Dropout(hidden_dropout_prob),

            nn.Linear(bert_hidden_size // 2, bert_hidden_size // 4),
            nn.LayerNorm(bert_hidden_size // 4),
            nn.GELU(),
            nn.Dropout(hidden_dropout_prob),

            nn.Linear(bert_hidden_size // 4, num_labels)
        )

    def forward(self, input_ids, attention_mask, labels=None):
        """
        前向传播

        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            labels: 标签（训练时提供）

        Returns:
            包含loss和logits的输出对象
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # 获取最后N层的CLS token
        hidden_states = outputs.hidden_states
        last_n = torch.stack(hidden_states[-self.num_cls_layers:])
        cls_tokens = last_n[:, :, 0, :]

        # 融合多层CLS token
        if self.use_layer_attention:
            # 注意力机制融合
            attn_scores = (cls_tokens * self.layer_query).sum(dim=-1)
            weights = torch.softmax(attn_scores, dim=0)
            weighted_cls = (cls_tokens * weights.unsqueeze(-1)).sum(dim=0)
        else:
            # 加权平均融合
            weights = torch.softmax(self.cls_weights, dim=0)
            weighted_cls = (cls_tokens * weights.view(-1, 1, 1)).sum(dim=0)

        weighted_cls = self.dropout(weighted_cls)
        logits = self.classifier(weighted_cls)

        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=self.label_smoothing)
            loss = loss_fct(logits, labels)

        return type('Output', (), {
            'loss': loss,
            'logits': logits
        })()
