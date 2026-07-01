"""BERT意图分类推理脚本"""
import torch
from transformers import BertModel, BertTokenizer
import json
from typing import Tuple
import time
import sys
from pathlib import Path

try:
    from .model import BertWithCustomHead
except ImportError:
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.core.model import BertWithCustomHead


class IntentPredictor:
    """意图预测器"""
    def __init__(self, model_path: str, label_mapping_path: str, base_model_path: str = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 加载标签映射
        with open(label_mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        self.label2id = mapping['label2id']
        self.id2label = {int(k): v for k, v in mapping['id2label'].items()}

        # 创建并加载模型
        if base_model_path is None:
            base_model_path = 'bert-base-chinese'

        bert = BertModel.from_pretrained(base_model_path)
        bert.config.output_hidden_states = True
        self.model = BertWithCustomHead(bert, len(self.label2id))

        # 加载权重
        state_dict = torch.load(f'{model_path}/model_state_dict.bin', map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model.to(self.device).eval()

    def predict(self, text: str) -> Tuple[str, float, float]:
        """预测文本意图"""
        inputs = self.tokenizer(text, max_length=512, padding='max_length',
                               truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
        inference_time = time.time() - start_time

        pred_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_id].item()
        return self.id2label[pred_id], confidence, inference_time


def main():
    """主函数"""
    # 路径配置 
    model_path = r'D:\intent_model_weight\bert-base-chinese-data2-bs8-ep40-lr1e5-weight-20260630\best_model'
    label_mapping_path = r'C:\Users\ddf\Desktop\zzc\code\agent-intent-classification\intent_model_train\data_bert_revise2\label_mapping.json'
    base_model_path = r'D:\intent_model_weight\bert-base-chinese'

    # 初始化预测器
    predictor = IntentPredictor(model_path, label_mapping_path, base_model_path)

    # 测试样例
    test_texts = [
        "国内租车用户: 我想查询订单信息",
        "国际租车用户: 怎么修改订单？",
        "国内租车用户: 一天多少钱？",
        "国际租车用户: 在哪里取车？",
        "新西兰租车"
    ]

    print("BERT意图分类推理测试")
    print("=" * 50)

    for text in test_texts:
        intent, confidence, inf_time = predictor.predict(text)
        print(f"文本: {text}")
        print(f"预测: {intent} (置信度: {confidence:.4f}, 耗时: {inf_time:.4f}s)")
        print("-" * 50)


if __name__ == "__main__":
    main()
