import os
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

#set api key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# 2. 載入並處理資料庫 (讀取 .txt 並轉化為向量)
loader = TextLoader("definitions.txt", encoding="utf-8")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=5)
texts = text_splitter.split_documents(documents)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_db = Chroma.from_documents(texts, embeddings)

# 3. 定義「自定義提示詞 (Custom Prompt)」
# 這是讓 AI 知道它必須標註來源的關鍵
template = """你是一個專業的助理。請根據下方提供的【參考資料】來回答問題。
如果答案是從參考資料中找到的，請在回答的最開頭加上「根據 RAG 辭典資料庫：」。
如果你在資料中找不到相關資訊，到網路上搜尋相關資訊，並加註來源是來自於網路搜尋。

【參考資料】：
{context}

使用者問題：{question}
正式回答："""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# 4. 初始化 Gemini 與 RAG 鏈結
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, tools=[{"google_search_retrieval": {}}])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_db.as_retriever(search_kwargs={"k": 1}), # 找最相關的一個
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# 5. 執行對話迴圈
print("Chatbot 已啟動！(輸入 'exit' 退出)")
while True:
    user_input = input("你：")
    if user_input.lower() == 'exit':
        break
        
    response = qa_chain.invoke(user_input)
    print(f"AI：{response['result']}\n")
