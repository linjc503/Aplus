# 多模态商品智能系统

基于 CLIP + ChromaDB + Qwen-Agent 的商品智能检索与客服系统。

## 功能特性

- **向量数据库**：使用 ChromaDB 存储商品图片和文本的向量表示
- **多模态编码**：使用 CLIP 模型将图片和文本编码为统一向量空间
- **智能检索**：支持文本搜索和图片搜索相似商品
- **零样本分类**：无需训练即可将商品图片分类到 5 个类别
- **智能客服**：基于 Qwen 大模型的商品问答助手

## 技术栈

- CLIP (ViT-B/32) - 多模态模型
- ChromaDB - 向量数据库
- FastAPI - Web 框架
- Qwen-Agent - 阿里云通义千问 Agent
- TailwindCSS - 前端样式

## 快速开始

### 1. 环境准备

```bash
# 安装依赖（如果需要）
pip install clip chromadb fastapi uvicorn torch torchvision pillow requests
pip install qwen-agent  # 可选，启用聊天功能需要
```

### 2. 配置

编辑 `app.py`，填写 API Key：

```python
DASHSCOPE_API_KEY = "your-api-key-here"  # 第27行
```

### 3. 启动服务

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 4. 访问

- 前端页面：http://localhost:8000
- API 文档：http://localhost:8000/docs

## 项目结构

```
项目/                    
├── 必做/                       
│   ├── vector_db/              # 向量数据库存储目录
│   └── build_vector_db.py      # 构建向量数据库的脚本
├── 选做/                       
│   └── app.py                  # 主应用脚本
├── chromadb存储结构.md         # ChromaDB 存储结构说明文档
├── README.md                   # 项目说明文档
├── 接口文档.md                 # 接口说明文档
└── 说明.md                     # 心得报告文档
```

## 功能模块

### 1. 文本录入
输入商品名称，自动编码存入向量数据库。

### 2. 图片录入
上传商品图片，可选添加描述文字。

### 3. 零样本分类
上传图片，自动识别类别：Topwear、Bottomwear、Accessories、Footwear、PersonalCare。

### 4. 相似性搜索
- 文本搜索：输入文字查找相似商品
- 图片搜索：上传图片查找相似商品

### 5. 智能客服
基于 Qwen 大模型的商品问答助手，可自动搜索商品回答问题。

## 数据说明

- 商品数据来自 `dataset1/styles.csv`
- 图片存储在 `dataset1/images/`
- 向量数据持久化在 `vector_db/` 目录