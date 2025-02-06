from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
import uvicorn
from rag_chain import rag_chain

app = FastAPI(
    title="笔记助手 API",
    version="1.1",
    description="基于 Obsidian 笔记的问答系统"
)

# CORS 中间件配置
app.add_middleware(
    CORSMiddleware,
    # 允许的源（开发环境下可以用 "*"，生产环境要具体指定）
    allow_origins=["*"],
    
    # 是否允许发送凭证（cookies等）
    allow_credentials=True,
    
    # 允许的 HTTP 方法
    allow_methods=["GET", "POST", "OPTIONS"],
    
    # 允许的 HTTP 请求头
    allow_headers=["*"],
    
    # 允许暴露的响应头
    expose_headers=["*"],
)

# 添加 RAG 链路由
add_routes(
    app,
    rag_chain,
    # path="/rag",
)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)