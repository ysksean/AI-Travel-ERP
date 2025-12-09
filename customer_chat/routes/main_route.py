import os
from flask import Blueprint, request, jsonify, render_template
import google.generativeai as genai
from dotenv import load_dotenv
from services.rag_service import rag_engine  # 위에서 만든 가짜 엔진 import

# .env 로드
load_dotenv()

bp = Blueprint('main', __name__)

# Gemini 설정
GENAI_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GENAI_KEY)
# 빠르고 저렴한 Flash 모델 사용
model = genai.GenerativeModel('gemini-2.5-flash-lite')


@bp.route('/')  # http://127.0.0.1:5000/ 접속 시
def home():
    return render_template('chat.html')


@bp.route('/chat', methods=['POST'])
def chat():
    # 1. 사용자 질문 받기
    data = request.json
    user_query = data.get('query', '')
    print(f"User Query: {user_query}")

    # 2. (Mock) RAG 엔진에게 정보 검색 요청
    # 지금은 무조건 '미야코지마 129만원' 정보가 나옴
    context = rag_engine.search(user_query)

    # 3. Gemini에게 프롬프트 전달
    system_prompt = f"""
    당신은 '캐리골프투어'의 친절한 상담원입니다.
    아래 [참고 정보]를 바탕으로 고객의 질문에 답해주세요.
    없는 내용은 지어내지 말고 "정보가 없습니다"라고 하세요.

    [참고 정보]
    {context}

    [고객 질문]
    {user_query}
    """

    # 4. 답변 생성 및 반환
    try:
        response = model.generate_content(system_prompt)
        return jsonify({
            "answer": response.text,
            "used_context": context  # 디버깅용 확인
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    pass