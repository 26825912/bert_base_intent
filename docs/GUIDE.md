# 使用指南

## 环境要求

- Python 3.10
- CUDA 13.0（GPU 加速）

## 快速开始

```bash
# 1. 安装 PyTorch（CUDA 13.0）
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 2. 安装其他依赖
pip install -r requirements.txt

# 2. 数据预处理
python -m src.core.data_preprocessing \
  --input data/raw_data.json \
  --output data/

# 3. 训练模型
python -m src.core.train

# 4. 启动 API
python -m src.api.api
```

## 数据格式

原始数据（JSON 格式）：

```json
[
  {
    "query": "我想查询订单信息",
    "label": "ORDER_QUERY"
  }
]
```

## 配置文件

编辑 `config/config.yaml` 修改配置：

```yaml
# 模型配置
model:
  base_model_name: "bert-base-chinese"
  num_cls_layers: 6
  hidden_dropout_prob: 0.1

# 训练配置
training:
  batch_size: 16
  num_epochs: 100
  learning_rate: 0.0003
  patience: 4

# 路径配置
paths:
  data_dir: "./data"
  output_dir: "./models"
```

## 核心命令

### 数据预处理

```bash
python -m src.core.data_preprocessing \
  --input data/raw_data.json \
  --output data/ \
  --train_ratio 0.8 \
  --val_ratio 0.1
```

### 模型训练

```bash
# 使用默认配置
python -m src.core.train

# 覆盖参数
python -m src.core.train \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --num_epochs 50
```

### API 测试

```bash
# 单条预测
curl -X POST "http://localhost:8085/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "我想查询订单"}'

# 批量预测
curl -X POST "http://localhost:8085/batch_predict" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["我想查询订单", "怎么修改订单"]}'
```

## 常见问题

**Q: 显存不足？**  
A: 减小 `batch_size` 到 8 或 4

**Q: 训练速度慢？**  
A: 确保使用 GPU，检查 `config.yaml` 中的 `cuda_visible_devices`

**Q: 找不到模型文件？**  
A: 首次运行会自动下载 BERT 模型，确保网络正常

**Q: API 无法访问？**  
A: 检查端口 8085 是否被占用

## 项目结构

```
bert_base_intent/
├── config/
│   └── config.yaml          # 配置文件
├── src/
│   ├── core/
│   │   ├── model.py         # 模型定义
│   │   ├── train.py         # 训练脚本
│   │   ├── inference.py     # 推理脚本
│   │   └── data_preprocessing.py  # 数据预处理
│   └── api/
│       └── api.py           # API 服务
└── tests/
    └── client.py            # API 测试客户端
```

## 更多信息

- 查看 API 文档: `http://localhost:8085/docs`
- 详细说明请参考主 [README.md](../README.md)
