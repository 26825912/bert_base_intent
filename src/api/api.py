"""
FastAPI 推理接口 - 意图分类单条推理
"""
import os

# 【重要】必须在导入 torch 之前设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import uvicorn
from typing import Optional, Dict
import logging
import json
import time
from transformers import BertModel, BertTokenizer

# 导入模型类
from src.core.model import BertWithCustomHead

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(title="意图分类推理服务", version="1.0.0")

# 全局预测器
predictor = None


class Request(BaseModel):
    """单条预测请求模型"""
    text: str
    return_probabilities: Optional[bool] = False


class BatchRequest(BaseModel):
    """批量预测请求模型"""
    texts: list[str]
    return_probabilities: Optional[bool] = False


class Response(BaseModel):
    """单条预测响应模型"""
    intent: str
    confidence: float
    probabilities: Optional[Dict[str, float]] = None
    inference_time: float


class BatchResponse(BaseModel):
    """批量预测响应模型"""
    results: list[Response]
    total_time: float
    average_time: float


class IntentPredictor:
    """意图预测器（简化版）"""

    def __init__(self, model_path: str, label_mapping_path: str, base_model_path: str = 'bert-base-chinese', device: str = 'cuda'):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")

        # 加载标签映射
        with open(label_mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        self.label2id = mapping['label2id']
        self.id2label = {int(k): v for k, v in mapping['id2label'].items()}

        # 创建并加载模型
        logger.info(f"加载基础模型: {base_model_path}")
        bert = BertModel.from_pretrained(base_model_path)
        bert.config.output_hidden_states = True
        self.model = BertWithCustomHead(bert, len(self.label2id))

        # 加载权重
        logger.info(f"加载训练权重: {model_path}/model_state_dict.bin")
        state_dict = torch.load(f'{model_path}/model_state_dict.bin', map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model.to(self.device).eval()

        logger.info("模型加载完成")

    def predict(self, text: str, return_probabilities: bool = False):
        """
        预测单个文本的意图

        Args:
            text: 输入文本
            return_probabilities: 是否返回所有类别的概率

        Returns:
            (预测意图, 置信度, 概率字典, 推理时间)
        """
        # 编码文本
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        # 预测并计时
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
        inference_time = time.time() - start_time

        # 获取预测结果
        predicted_class_id = torch.argmax(probabilities, dim=1).item()
        predicted_intent = self.id2label[predicted_class_id]
        confidence = probabilities[0][predicted_class_id].item()

        if return_probabilities:
            # 返回所有类别的概率
            prob_dict = {}
            for class_id, prob in enumerate(probabilities[0]):
                prob_dict[self.id2label[class_id]] = prob.item()
            return predicted_intent, confidence, prob_dict, inference_time

        return predicted_intent, confidence, {}, inference_time


def get_predictor():
    """获取预测器（单例模式）"""
    global predictor
    if predictor is None:
        # 路径配置 - 需要根据实际部署环境修改
        model_path = './models/best_model'
        label_mapping_path = './data/label_mapping.json'
        base_model_path = 'bert-base-chinese'

        predictor = IntentPredictor(model_path, label_mapping_path, base_model_path)
    return predictor


@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    logger.info("服务启动中，正在加载模型...")
    get_predictor()
    logger.info("服务启动完成，模型已加载到显存")


@app.post("/predict", response_model=Response)
async def predict(request: Request):
    """
    单条文本意图预测

    Args:
        request: 包含文本的请求体

    Returns:
        预测结果（意图、置信度、概率分布、推理时间）
    """
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="文本不能为空")

        pred = get_predictor()

        # 预测
        intent, confidence, probs, inference_time = pred.predict(
            request.text,
            return_probabilities=request.return_probabilities
        )

        return Response(
            intent=intent,
            confidence=confidence,
            probabilities=probs if request.return_probabilities else None,
            inference_time=inference_time
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"预测失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.post("/batch_predict", response_model=BatchResponse)
async def batch_predict(request: BatchRequest):
    """
    批量文本意图预测

    Args:
        request: 包含文本列表的请求体

    Returns:
        批量预测结果
    """
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="文本列表不能为空")

        if len(request.texts) > 100:
            raise HTTPException(status_code=400, detail="单次批量预测最多支持100条文本")

        pred = get_predictor()
        results = []
        total_start_time = time.time()

        for text in request.texts:
            if not text or not text.strip():
                # 空文本返回默认结果
                results.append(Response(
                    intent="UNKNOWN",
                    confidence=0.0,
                    probabilities=None,
                    inference_time=0.0
                ))
                continue

            intent, confidence, probs, inference_time = pred.predict(
                text,
                return_probabilities=request.return_probabilities
            )

            results.append(Response(
                intent=intent,
                confidence=confidence,
                probabilities=probs if request.return_probabilities else None,
                inference_time=inference_time
            ))

        total_time = time.time() - total_start_time
        average_time = total_time / len(request.texts) if request.texts else 0

        return BatchResponse(
            results=results,
            total_time=total_time,
            average_time=average_time
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量预测失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"批量预测失败: {str(e)}")


@app.get("/")
async def root():
    """根路径 - 服务状态"""
    return {"message": "意图分类推理服务运行中", "docs": "/docs", "predict": "/predict"}


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8085)
