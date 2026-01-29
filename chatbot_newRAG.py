import os
# import google.generativeai as genai # Deprecated, using google.genai
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from google import genai
from google.genai import types

#set api key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# 2. 載入並處理資料庫 (讀取 .txt 並轉化為向量)
# 這裡沿用 LangChain 的 Document Loader 和 Splitter，因為它們處理文本很方便
try:
    loader = TextLoader("definitions.txt", encoding="utf-8")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=5)
    texts = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_db = Chroma.from_documents(texts, embeddings)
except Exception as e:
    print(f"資料庫初始化失敗: {e}")
    exit(1)

# 3. 定義「自定義提示詞 (Custom Prompt)」
template = """ 請根據下方提供的【參考資料】來回答問題。
如果答案是從參考資料中找到的，請在回答的最開頭加上「根據 RAG 辭典資料庫：」。
如果你在資料中找不到相關資訊，到網路上搜尋相關資訊，註明來源是來自於網路搜尋。

【參考資料】：
{context}

使用者問題：{question}
正式回答："""

# 4. 初始化 Gemini Client (使用 google.genai SDK 以支援 Google Search Grounding)
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))



# 5. 執行對話迴圈
print("Chatbot (New RAG) 已啟動！(輸入 'exit' 退出)")
while True:
    user_input = input("你：")
    if user_input.lower() == 'exit':
        break
        
    try:
        # 手動檢索 (Retrieval)
        # k=1 找最相關的一個
        docs = vector_db.similarity_search(user_input, k=1)
        
        #context處理
        content_list = []
        if docs:
            for doc in docs:
                content_list.append(doc.page_content)
            context_text = "\n\n".join(content_list)
        else:
            context_text="無相關資料"

        
        final_prompt = template.format(context=context_text, question=user_input)
        
        #選擇使用模型
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=final_prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.3
            )
        )
        print(f"AI：{response.text}\n")
        
    except Exception as e:
        print(f"發生錯誤：{e}\n")
