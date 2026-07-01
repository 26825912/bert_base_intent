# BERT 意图分类模型

基于 BERT-base-chinese 的意图分类模型训练和部署框架。

## 项目特点

- 完整的训练流程：数据预处理、模型训练、验证测试
- 生产级 API 服务：FastAPI 实现，支持单条/批量预测
- 配置驱动：统一的 YAML 配置文件，支持命令行覆盖
- 早停机制：自动保存最佳模型，防止过拟合

## 项目结构

```
bert_base_intent/
├── config/          # 配置文件
├── src/
│   ├── core/       # 训练和推理模块
│   └── api/        # FastAPI 服务
├── tests/          # 测试脚本
└── docs/           # 使用文档
```

## 环境要求

- Python 3.10
- CUDA 13.0（GPU 加速）

## 快速开始

### 1. 安装依赖

```bash
# 安装 PyTorch（CUDA 13.0，也可使用其他能调用 GPU 的 CUDA 版本）
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 安装其他依赖（包含 transformers==5.8.1）
pip install -r requirements.txt
```

> **注意**：PyTorch 必须使用上述命令单独安装。推荐 CUDA 13.0，也可使用其他能正常调用 GPU 的 CUDA 版本。transformers 固定使用 5.8.1 版本。

### 2. 数据预处理

准备 JSON 格式数据：

```json
[
  {"query": "我想查询订单信息", "label": "ORDER_QUERY"}
]
```

运行预处理：

```bash
python -m src.core.data_preprocessing \
  --input data/raw_data.json \
  --output data/
```

### 3. 训练模型

```bash
# 使用默认配置
python -m src.core.train

# 自定义参数
python -m src.core.train --batch_size 32 --learning_rate 0.0001
```

配置文件位于 `config/config.yaml`，可修改模型参数、训练超参数等。

### 4. 启动 API 服务

```bash
python -m src.api.api
```

服务启动后访问 `http://localhost:8085/docs` 查看 API 文档。

### 5. API 调用示例

**单条预测**：
```bash
curl -X POST "http://localhost:8085/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "我想查询订单"}'
```

**批量预测**：
```bash
curl -X POST "http://localhost:8085/batch_predict" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["我想查询订单", "怎么修改订单"]}'
```

## 模型架构

- **基础模型**: BERT-base-chinese (768维)
- **特征融合**: 多层 CLS token 加权融合
- **分类头**: 768 → 384 → 192 → num_labels
- **优化器**: AdamW + 层差异化学习率
- **损失函数**: CrossEntropyLoss + Label Smoothing

## 主要依赖

- PyTorch 2.0+ (CUDA 13.0)
- Transformers 5.8.1
- FastAPI
- scikit-learn

## 最近更新 (2026-06-30)

✨ **核心优化完成**：

- ✅ 配置管理系统（config.yaml）
- ✅ 命令行参数支持
- ✅ 数据预处理工具
- ✅ 批量推理接口
- ✅ 完善的错误处理

## 文档

- [使用指南](docs/GUIDE.md) - 详细使用说明和常见问题

## License

MIT License
