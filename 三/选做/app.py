from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
import clip
import chromadb
import torch
from PIL import Image
import io
import uuid
import requests
from typing import List, Dict, Any, Optional

# 初始化FastAPI应用
app = FastAPI(title="多模态商品智能系统")

# 初始化CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 连接ChromaDB
client = chromadb.PersistentClient(path="./vector_db")
collection = client.get_or_create_collection(
    name="products",
    metadata={"hnsw:space": "cosine"}
)

# Qwen Agent配置
DASHSCOPE_API_KEY = "Your API KEY"  # 用户需要填写自己的API Key

# 零样本分类的类别
CATEGORIES = ["Topwear", "Bottomwear", "Accessories", "Footwear", "PersonalCare"]

# 生成新的商品ID
def generate_product_id() -> str:
    """生成唯一的商品ID"""
    # 这个uuid是给用户添加进来的商品用的，完全随机，取八位
    return str(uuid.uuid4())[:8]

# 文本编码函数
def encode_text(text: str) -> List[float]:
    """将文本编码为向量"""
    with torch.no_grad():
        text_features = model.encode_text(clip.tokenize([text]).to(device))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features[0].cpu().numpy().tolist()

# 图片编码函数
def encode_image(image: Image.Image) -> List[float]:
    """将图片编码为向量"""
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features[0].cpu().numpy().tolist()




# search_products工具函数
def search_products(query: str) -> str:
    """搜索商品的Tool，调用本地/search_text接口"""
    '''
    response:
    {
  "query": "红色上衣",
  "results": [
    {
      "id": "1001_image",
      "similarity": 0.92,
      "metadata": {
        "product_id": "1001",
        "description": "红色时尚上衣"
      }
    }
  ]
}
    '''
    try:
        response = requests.post(
            "http://localhost:8000/search_text",
            # 这个query就是用户输入的问题
            json={"text": query},
            # 十秒种没有返回答复，就自动停止等待
            timeout=10
        )   
        # 200 表示成功连接并搜索
        if response.status_code == 200:
            data = response.json()
            # results是一个列表，如果没东西get就返回一个空[]，防止报错
            results = data.get("results", [])
            if not results:
                return "未找到相关商品"
            
            product_list = []
            for item in results:
                metadata = item.get("metadata", {})
                product_id = metadata.get("product_id", "unknown")
                product_type = metadata.get("type", "unknown")
                text = metadata.get("text", "")
                description = metadata.get("description", "")
                
                if text:
                    product_list.append(f"商品ID: {product_id}, 类型: {product_type}, 描述: {text}")
                elif description:
                    product_list.append(f"商品ID: {product_id}, 类型: {product_type}, 描述: {description}")
                else:
                    product_list.append(f"商品ID: {product_id}, 类型: {product_type}")
            
            return "\n".join(product_list)
        else:
            return f"搜索失败: {response.status_code}"
    except Exception as e:
        return f"搜索出错: {str(e)}"



# 初始化Qwen Agent
llm = None
agent = None

def init_qwen_agent():
    """初始化Qwen Agent"""
    global llm, agent
    
    if not DASHSCOPE_API_KEY:
        print("警告: DASHSCOPE_API_KEY未设置，聊天功能将不可用")
        return
    
    try:
        from qwen_agent.tools.base import BaseTool
        from qwen_agent.agents import ReActChat 
        
        # 定义商品搜索工具（不需要 @register_tool 装饰器）
        class ProductSearchTool(BaseTool):
            name = "search_products"
            description = "搜索商品信息，当需要查找商品时使用此工具"
            
            def call(self, query: str, **kwargs) -> str:   # 注意加上 **kwargs
                return search_products(query)
        
        # 初始化LLM配置
        llm = {
            "model": "qwen-max",
            "api_key": DASHSCOPE_API_KEY
        }
        
        # 关键：直接传入工具实例，而不是字符串
        agent = ReActChat(
            llm=llm,
            function_list=[ProductSearchTool()],   # 传实例，不是字符串
            system_message="你是一个商品客服助手，可以帮助用户搜索和推荐商品。"
        )
        
        print("Qwen Agent初始化成功")
    except ImportError as e:
        print(f"警告: qwen_agent模块未安装，聊天功能将不可用: {e}")
    except Exception as e:
        print(f"警告: Qwen Agent初始化失败: {e}")

# 启动时初始化Agent
init_qwen_agent()




# 根路由 - 前端页面
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """返回前端页面"""
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>多模态商品智能系统</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body { font-family: Arial, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; }
            .container { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .card { border: 1px solid #ddd; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            input[type="text"], input[type="file"] { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; }
            button { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background-color: #45a049; }
            .result { margin-top: 20px; padding: 10px; background-color: #f0f0f0; border-radius: 4px; max-height: 200px; overflow-y: auto; }
            
            /* 聊天区域样式 */
            .chat-container { 
                grid-column: 1 / -1; 
                background: white; 
                border-radius: 12px; 
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                padding: 20px;
                margin-top: 20px;
            }
            .chat-header { font-size: 1.5rem; font-weight: bold; margin-bottom: 15px; color: #333; }
            .chat-messages { 
                height: 400px; 
                overflow-y: auto; 
                border: 1px solid #e0e0e0; 
                border-radius: 8px; 
                padding: 15px;
                margin-bottom: 15px;
                background-color: #f9f9f9;
            }
            .chat-message { margin-bottom: 15px; display: flex; }
            .chat-message.user { justify-content: flex-end; }
            .chat-message.ai { justify-content: flex-start; }
            .message-bubble { 
                max-width: 70%; 
                padding: 12px 16px; 
                border-radius: 12px; 
                line-height: 1.5;
            }
            .chat-message.user .message-bubble { 
                background-color: #4CAF50; 
                color: white; 
                border-bottom-right-radius: 4px;
            }
            .chat-message.ai .message-bubble { 
                background-color: white; 
                color: #333; 
                border: 1px solid #ddd;
                border-bottom-left-radius: 4px;
            }
            .chat-input-area { display: flex; gap: 10px; }
            .chat-input-area input { flex: 1; margin: 0; }
            .chat-input-area button { background-color: #2196F3; }
            .chat-input-area button:hover { background-color: #1976D2; }
        </style>
    </head>
    <body>
        <h1 class="text-3xl font-bold mb-6">多模态商品智能系统</h1>
        
        <div class="container">
            <!-- 文本录入区域 -->
            <div class="card">
                <h2 class="text-xl font-semibold mb-4">文本录入</h2>
                <input type="text" id="textInput" placeholder="请输入商品名称">
                <button onclick="addText()">添加文本</button>
                <div id="textResult" class="result"></div>
            </div>
            
            <!-- 图片录入区域 -->
            <div class="card">
                <h2 class="text-xl font-semibold mb-4">图片录入</h2>
                <input type="file" id="imageInput" accept="image/*">
                <input type="text" id="imageDesc" placeholder="可选：商品描述">
                <button onclick="addImage()">添加图片</button>
                <div id="imageResult" class="result"></div>
            </div>
            
            <!-- 零样本分类区域 -->
            <div class="card">
                <h2 class="text-xl font-semibold mb-4">零样本分类</h2>
                <input type="file" id="classifyInput" accept="image/*">
                <button onclick="classifyImage()">分类图片</button>
                <div id="classifyResult" class="result"></div>
            </div>
            
            <!-- 聊天区域 -->
            <div class="chat-container">
                <div class="chat-header">商品客服助手</div>
                <div class="chat-messages" id="chatMessages"></div>
                <div class="chat-input-area">
                    <input type="text" id="chatInput" placeholder="请输入您的问题..." onkeypress="handleChatKeyPress(event)">
                    <button onclick="sendMessage()">发送</button>
                </div>
            </div>
        </div>
        
        <script>
            // 添加文本
            async function addText() {
                const text = document.getElementById('textInput').value;
                if (!text) { alert('请输入商品名称'); return; }
                
                const response = await fetch('/add_text', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });
                
                const result = await response.json();
                document.getElementById('textResult').innerText = JSON.stringify(result, null, 2);
            }
            
            // 添加图片
            async function addImage() {
                const fileInput = document.getElementById('imageInput');
                const descInput = document.getElementById('imageDesc');
                
                if (!fileInput.files[0]) { alert('请选择图片'); return; }
                
                const formData = new FormData();
                formData.append('image', fileInput.files[0]);
                formData.append('description', descInput.value);
                
                const response = await fetch('/add_image', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                document.getElementById('imageResult').innerText = JSON.stringify(result, null, 2);
            }
            
            // 分类图片
            async function classifyImage() {
                const fileInput = document.getElementById('classifyInput');
                
                if (!fileInput.files[0]) { alert('请选择图片'); return; }
                
                const formData = new FormData();
                formData.append('image', fileInput.files[0]);
                
                const response = await fetch('/classify_image', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                document.getElementById('classifyResult').innerText = JSON.stringify(result, null, 2);
            }
            
            // 添加消息到聊天区域
            function addMessage(message, isUser) {
                const messagesContainer = document.getElementById('chatMessages');
                const messageDiv = document.createElement('div');
                messageDiv.className = 'chat-message ' + (isUser ? 'user' : 'ai');
                messageDiv.innerHTML = '<div class="message-bubble">' + message.replace(/\\n/g, '<br>') + '</div>';
                messagesContainer.appendChild(messageDiv);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
            
            // 发送消息
            async function sendMessage() {
                const input = document.getElementById('chatInput');
                const message = input.value.trim();
                if (!message) return;
                
                addMessage(message, true);
                input.value = '';
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message })
                    });
                    
                    const result = await response.json();
                    addMessage(result.reply || result.error || '处理失败', false);
                } catch (error) {
                    addMessage('发送失败: ' + error.message, false);
                }
            }
            
            // 处理回车键
            function handleChatKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }
            
            // 页面加载完成后滚动到最新消息
            window.onload = function() {
                const messagesContainer = document.getElementById('chatMessages');
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            };
        </script>
    </body>
    </html>
    """
    return html_content


# 添加文本接口 X
@app.post("/add_text")
async def add_text(text: str) -> Dict[str, Any]:
    """添加文本到向量数据库"""
    product_id = generate_product_id()
    text_vector = encode_text(text)
    
    collection.add(
        ids=[f"{product_id}_text"],
        embeddings=[text_vector],
        metadatas=[{"type": "text", "product_id": product_id, "text": text}]
    )
    
    return {"product_id": product_id, "text": text}

# 添加图片接口 X
@app.post("/add_image")
async def add_image(
    image: UploadFile = File(...),
    description: Optional[str] = Form(None)
) -> Dict[str, Any]:
    """添加图片到向量数据库"""
    product_id = generate_product_id()
    
    image_data = await image.read()
    img = Image.open(io.BytesIO(image_data))
    
    image_vector = encode_image(img)
    
    collection.add(
        ids=[f"{product_id}_image"],
        embeddings=[image_vector],
        metadatas=[{"type": "image", "product_id": product_id, "description": description}]
    )
    
    if description:
        text_vector = encode_text(description)
        collection.add(
            ids=[f"{product_id}_text"],
            embeddings=[text_vector],
            metadatas=[{"type": "text", "product_id": product_id, "text": description}]
        )
    
    return {"product_id": product_id, "description": description}

# 文本搜索接口
@app.post("/search_text")
async def search_text(text: str) -> Dict[str, Any]:
    """搜索相似文本"""
    text_vector = encode_text(text)
    
    results = collection.query(
        query_embeddings=[text_vector],
        n_results=5,
        where={"type": {"$in": ["text", "image"]}}
    )
    
    search_results = []
    for i, (id_, distance, metadata) in enumerate(zip(
        results['ids'][0],
        results['distances'][0],
        results['metadatas'][0]
    )):
        search_results.append({
            "id": id_,
            "similarity": 1 - distance,
            "metadata": metadata
        })
    
    return {"query": text, "results": search_results}

# 图片搜索接口 X
@app.post("/search_image")
async def search_image(image: UploadFile = File(...)) -> Dict[str, Any]:
    """搜索相似图片"""
    image_data = await image.read()
    img = Image.open(io.BytesIO(image_data))
    
    image_vector = encode_image(img)
    
    results = collection.query(
        query_embeddings=[image_vector],
        n_results=5,
        where={"type": {"$in": ["text", "image"]}}
    )
    
    search_results = []
    for i, (id_, distance, metadata) in enumerate(zip(
        results['ids'][0],
        results['distances'][0],
        results['metadatas'][0]
    )):
        search_results.append({
            "id": id_,
            "similarity": 1 - distance,
            "metadata": metadata
        })
    
    return {"results": search_results}

# 零样本分类接口
@app.post("/classify_image")
async def classify_image(image: UploadFile = File(...)) -> Dict[str, Any]:
    """零样本分类图片"""
    image_data = await image.read()
    img = Image.open(io.BytesIO(image_data))
    
    image_vector = encode_image(img)
    
    category_vectors = []
    for category in CATEGORIES:
        category_vector = encode_text(category)
        category_vectors.append(category_vector)
    
    import numpy as np
    image_vector_np = np.array(image_vector)
    similarities = []
    
    for category, vec in zip(CATEGORIES, category_vectors):
        vec_np = np.array(vec)
        similarity = np.dot(image_vector_np, vec_np)
        similarities.append((category, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_category, top_confidence = similarities[0]
    
    return {
        "category": top_category,
        "confidence": float(top_confidence),
        "all_categories": [
            {"category": cat, "confidence": float(conf)} 
            for cat, conf in similarities
        ]
    }

# 聊天接口
@app.post("/chat")
async def chat(request: Request) -> Dict[str, Any]:
    """商品客服聊天接口"""
    global agent
    
    # 手动解析 JSON body
    try:
        body = await request.json()
        message = body.get("message", "")
    except Exception:
        return {"reply": "请求格式错误"}
    
    if not message:
        return {"reply": "消息不能为空"}
    
    if not agent:
        return {"reply": "抱歉，Qwen Agent未初始化。请检查DASHSCOPE_API_KEY是否正确设置。"}
    
    try:
        # 遍历生成器获取响应
        responses = []
        for response in agent.run([{"role": "user", "content": message}]):
            responses.append(response)
        
        if responses:
            final_response = responses[-1]
            if isinstance(final_response, list) and len(final_response) > 0:
                reply = final_response[-1].get("content", str(final_response))
            elif isinstance(final_response, dict):
                reply = final_response.get("content", str(final_response))
            else:
                reply = str(final_response)
        else:
            reply = "抱歉，我没有收到回复。"
        
        return {"reply": reply}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"reply": f"处理出错: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)