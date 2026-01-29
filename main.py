import os
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA



load_dotenv()
#genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)



# 初始化模型
try:
    model = genai.GenerativeModel('gemini-2.0-flash')
except Exception as e:
    print(f"模型初始化失敗: {e}")
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    if model is None:
        return jsonify({'error': '模型尚未初始化'}), 500

    user_message = request.json.get('message')
    
    try:
        response = model.generate_content(user_message)
        return jsonify({'response': response.text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001)
