
import os
from dotenv import load_dotenv

#配置notebook走代理
os.environ['https_proxy']='http://127.0.0.1:7890'
os.environ['http_proxy']='http://127.0.0.1:7890'
# os.environ['al1_proxy']='socks5://127.0.0.1:7890'

#配置USER_AGENT
os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

#配置api_key
load_dotenv()
chat_api_key = os.getenv("QWEN_API_KEY")

#配置langsmith
LANGSMITH_TRACING=True
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "RAG-document-project"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import List, TypedDict
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders.obsidian import ObsidianLoader
from langchain_community.document_loaders.notion import NotionDirectoryLoader
#全局debug模式
from langchain_core.globals import set_debug  


loader = ObsidianLoader(r"D:\Obsidian Vault")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter()
splitted_docs = text_splitter.split_documents(docs)

# vectordb = Chroma.from_documents(documents=splitted_docs,
#                                     #  embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#                                  embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh"),
#                                 #  persist_directory='./chroma_db'
#                                 )
# vectordb = Chroma(persist_directory='./chroma_db',
#        embedding_function=HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
# )
persist_directory = './chroma_db'
if os.path.exists(persist_directory) and os.listdir(persist_directory):
    print("加载已存在的向量数据库")
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")
    )
    
else:
    print("创建新的向量数据库")
    vectordb = Chroma.from_documents(
        documents=splitted_docs,
        embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh"),
        persist_directory=persist_directory
    )

retriever = vectordb.as_retriever()

template = """
你是一个专业的笔记助手，请基于提供的笔记内容回答问题。

笔记内容：
{notes}

请注意：
1. 如果笔记内容中包含相关信息，请详细解答
2. 如果笔记内容中没有相关信息，请明确说明
3. 回答时要考虑笔记的结构和上下文
4. 如果涉及多个笔记片段，请综合分析后回答

问题：{question}

请给出清晰、准确的回答：

"""
prompt = ChatPromptTemplate.from_template(template)

llm = BaseChatOpenAI(
    model='qwen-plus',
    openai_api_key=chat_api_key,
    openai_api_base='https://dashscope.aliyuncs.com/compatible-mode/v1',
    max_tokens=1024,
    temperature=0.2
)

rag_chain = (
    {"notes": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)




