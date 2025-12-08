import sys
import os

try:
    import google.generativeai as genai
except ImportError:
    genai = None
    print("Warning: google-generativeai module not found.")

from flask import Flask, render_template, request, jsonify

# Manual .env loader since python-dotenv might be missing
def load_env_manual(filepath):
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        key, value = line.split('=', 1)
                        value = value.strip()
                        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                            value = value[1:-1]
                        os.environ[key.strip()] = value
            print(f"Successfully loaded .env from {filepath}")
        else:
            print("No .env file found.")
    except Exception as e:
        print(f"Error loading .env file: {e}")

# Load environment variables
load_env_manual(os.path.join(os.path.dirname(__file__), '.env'))

# Add sibling directory to sys.path to import from travel/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'travel', 'flask_web')))

try:
    from services.rag_service import rag_engine
    print("Successfully imported rag_engine")
except ImportError as e:
    print(f"Error importing rag_engine: {e}")
    rag_engine = None

app = Flask(__name__)

# Normalize API Key
if not os.getenv('GOOGLE_API_KEY') and os.getenv('MY_API_KEY'):
    os.environ['GOOGLE_API_KEY'] = os.getenv('MY_API_KEY')

# Configure Google Gemini API
if genai:
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
    else:
        print("Warning: GOOGLE_API_KEY not found in environment variables")

# Initialize RAG Engine with dummy data if needed (or assume it loads its own)
# The prompt asked to "Call rag_engine.add_knowledge(['mock data list...'])"
if rag_engine:
    mock_knowledge = [
        "홋카이도 4일 상품은 680,000원이며 온천 호텔 숙박이 포함됩니다.",
        "튀르키예 일주 8~10일 상품은 1,780,000원이며 전일정 5성급 호텔입니다.",
        "이탈리아 일주 8/9일 상품은 2,100,000원부터 시작합니다.",
        "시드니 5~8일 상품은 354,700원의 특가로 제공됩니다.",
        "다낭/호이안 4/5일 상품은 399,000원이며 바나힐 관광이 포함됩니다.",
        "코타키나발루 5일 상품은 420,000원으로 반딧불 투어가 인기입니다."
    ]
    rag_engine.add_knowledge(mock_knowledge)

# ... (rest of mock_data same as before, skipping for brevity in replacement tool if possible, but replace_file_content replaces chunk)
# I need to match the chunk exactly. The previous start line was 4 (import genai).
# I will use a smaller chunk for the top part and another for the chat function?
# No, replace_file_content works on a single contiguous block.
# I will replace the top imports first.

mock_data = {
    "hero_slides": [
        {
            "title": "사이판 5,6일 #켄싱턴호텔 #오션뷰\n호캉스 #1일\n2식/3식 호텔식",
            "hashtags": ["#온천호텔", "#교토", "#오사카", "#유니버셜스튜디오"],
            "bg_image": "https://images.unsplash.com/photo-1540206351-d6465b3ac5c1?ixlib=rb-4.0.3&auto=format&fit=crop&w=1600&q=80",
            "cards": [
                {
                    "title": "홋카이도 4일 #호텔 확정",
                    "original_price": "800,000",
                    "price": "680,000",
                    "discount": "15%",
                    "image": "https://images.unsplash.com/photo-1542051841857-5f90071e7989?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
                    "tags": ["#일본", "#온천"]
                },
                {
                    "title": "도쿄 3일 #시내관광",
                    "original_price": "600,000",
                    "price": "450,000",
                    "discount": "25%",
                    "image": "https://images.unsplash.com/photo-1503899036084-c55cdd92da26?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
                    "tags": ["#도쿄", "#쇼핑"]
                }
            ]
        },
        {
            "title": "다낭/호이안 4/5일 #바나힐\n#골든브릿지 #콩카페",
            "hashtags": ["#베트남", "#다낭", "#가족여행", "#휴양"],
            "bg_image": "https://images.unsplash.com/photo-1552465011-b4e21bf6e79a?ixlib=rb-4.0.3&auto=format&fit=crop&w=1600&q=80",
            "cards": [
                {
                    "title": "다낭 4일 #풀빌라",
                    "original_price": "900,000",
                    "price": "750,000",
                    "discount": "16%",
                    "image": "https://images.unsplash.com/photo-1565035010268-a3816f98589a?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
                    "tags": ["#휴양", "#수영장"]
                },
                {
                    "title": "나트랑 5일 #빈펄랜드",
                    "original_price": "850,000",
                    "price": "690,000",
                    "discount": "18%",
                    "image": "https://images.unsplash.com/photo-1565636291755-72a3707e5b92?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
                    "tags": ["#나트랑", "#테마파크"]
                }
            ]
        },
        {
            "title": "유럽의 낭만, 이탈리아 일주\n8/9일 #가성비여행",
            "hashtags": ["#유럽", "#이탈리아", "#로마", "#피렌체"],
            "bg_image": "https://images.unsplash.com/photo-1523906834658-6e24ef2386f9?ixlib=rb-4.0.3&auto=format&fit=crop&w=1600&q=80",
            "cards": [
                {
                    "title": "이탈리아 9일 #완전일주",
                    "original_price": "2,500,000",
                    "price": "2,100,000",
                    "discount": "16%",
                    "image": "https://images.unsplash.com/photo-1516483638261-f4dbaf036963?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
                    "tags": ["#역사", "#문화"]
                },
                {
                    "title": "스위스/이탈리아 10일",
                    "original_price": "3,200,000",
                    "price": "2,890,000",
                    "discount": "10%",
                    "image": "https://images.unsplash.com/photo-1527668752968-14dc70a27c95?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
                    "tags": ["#알프스", "#자연"]
                }
            ]
        }
    ],
    "icons": [
        {"label": "골프여행", "icon": "fa-solid fa-golf-ball-tee"},
        {"label": "허니문", "icon": "fa-solid fa-heart"},
        {"label": "휴양지", "icon": "fa-solid fa-umbrella-beach"},
        {"label": "동남아 여행", "icon": "fa-brands fa-youtube"},
        {"label": "패키지", "icon": "fa-solid fa-suitcase"},
        {"label": "크루즈", "icon": "fa-solid fa-ship"},
        {"label": "해외숙소", "icon": "fa-solid fa-hotel"},
        {"label": "항공예약", "icon": "fa-solid fa-plane"},
        {"label": "여행의 발견", "icon": "fa-brands fa-instagram"},
        {"label": "여행 LIVE", "icon": "fa-solid fa-life-ring"}
    ],
    "products_a": [
        {
            "title": "홋카이도 4일 #호텔 확정 #온천 호텔 숙박 #오타루 산책",
            "original_price": "800,000",
            "price": "680,000",
            "discount": "15%",
            "image": "https://images.unsplash.com/photo-1542051841857-5f90071e7989?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
            "tags": ["#일본", "#온천", "#가족여행"]
        },
        {
            "title": "튀르키예(터키) 일주 8~10일 #가성비 여행 #터키국내선1회 #터키음식3대",
            "original_price": "2,000,000",
            "price": "1,780,000",
            "discount": "11%",
            "image": "https://images.unsplash.com/photo-1524231757912-21f4fe3a7200?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
            "tags": ["#터키", "#역사", "#문화"]
        },
        {
            "title": "이탈리아 일주 8/9일 #가성비여행",
            "original_price": "2,100,000",
            "price": "1,799,000",
            "discount": "14%",
            "image": "https://images.unsplash.com/photo-1523906834658-6e24ef2386f9?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
            "tags": ["#유럽", "#이탈리아", "#낭만"]
        },
        {
            "title": "시드니 5~8일 #뜨거운 여름에 만나는 시원한 조개꽃! #블루마운틴 #포트",
            "original_price": "450,000",
            "price": "354,700",
            "discount": "21%",
            "image": "https://images.unsplash.com/photo-1506973035872-a4ec16b8e8d9?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
            "tags": ["#호주", "#시드니", "#자연"]
        }
    ],
    "promo": {
        "title": "매일이 즐겁고 풍요로운 동남아의 지상 낙원으로",
        "keywords": ["여유있는 힐링, 일본", "동남아의 지상낙원료", "여행 LIVE", "생생한 정보"],
        "bg_image": "https://images.unsplash.com/photo-1537996194471-e657df975ab4?ixlib=rb-4.0.3&auto=format&fit=crop&w=1600&q=80",
        "card": {
             "title": "튀르키예(터키) 일주 8~10일 #가성비 여행 #터키국내선...",
             "desc": "전일정 5성급호텔 숙박, 밸리댄스 포함, 사프란볼루 등 관광 포함, 알차게 다녀올 수 있는 상품입니다.",
             "original_price": "2,000,000",
             "price": "1,780,000",
             "discount": "11%",
             "image": "https://images.unsplash.com/photo-1527838832700-5059252407fa?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80"
        }
    },
    "products_b": [
        {
            "title": "푸켓 4일 #풀빌라 확정 #요트 투어 #스파 마사지",
            "original_price": "900,000",
            "price": "680,000",
            "discount": "24%",
            "image": "https://images.unsplash.com/photo-1589394815804-964ed0be2eb5?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
            "tags": ["#태국", "#휴양", "#풀빌라"]
        },
        {
            "title": "다낭/호이안 4/5일 #바나힐 #골든브릿지 #콩카페",
            "original_price": "500,000",
            "price": "399,000",
            "discount": "20%",
            "image": "https://images.unsplash.com/photo-1552465011-b4e21bf6e79a?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
            "tags": ["#베트남", "#다낭", "#가족"]
        },
        {
            "title": "보라카이 4/5일 #화이트비치 #세일링보트 #호핑투어",
            "original_price": "600,000",
            "price": "450,000",
            "discount": "25%",
            "image": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
            "tags": ["#필리핀", "#보라카이", "#바다"]
        },
        {
            "title": "코타키나발루 5일 #반딧불투어 #선셋 #호핑투어",
            "original_price": "550,000",
            "price": "420,000",
            "discount": "23%",
            "image": "https://images.unsplash.com/photo-1573455494060-c5595004fb6c?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80",
            "tags": ["#말레이시아", "#석양", "#자연"]
        }
    ]
}

@app.route('/')
def index():
    return render_template('index.html', data=mock_data)

@app.route('/chat', methods=['POST'])
def chat():
    print("Chat endpoint called")
    data = request.json
    user_message = data.get('message')
    print(f"User message: {user_message}")

    if not user_message:
        return jsonify({'reply': '메시지를 입력해주세요.'}), 400

    try:
        # 1. RAG Retrieval
        context = ""
        if rag_engine:
            print("Searching in RAG engine...")
            context = rag_engine.search(user_message)
            print(f"Retrieved context: {context}")
        else:
            print("RAG engine not available")
        
        # 2. Gemini Generation
        if genai and os.getenv('GOOGLE_API_KEY'):
            model_name = os.getenv('MODEL_NAME', 'gemini-1.5-flash')
            try:
                model = genai.GenerativeModel(model_name)
                print(f"Using model: {model_name}")
            except Exception:
                model = genai.GenerativeModel('gemini-pro')
                print("Fallback to model: gemini-pro")
            prompt = f"""
            You are a helpful travel agent for 'AI Hanatour'.
            Use the following context to answer the user's question politely and professionally in Korean.
            If the context doesn't have the answer, answer based on general travel knowledge but mention you are not sure about specific product details.
            
            Context:
            {context}
            
            User Question:
            {user_message}
            
            Answer:
            """
            
            print("Generating response with Gemini...")
            response = model.generate_content(prompt)
            reply = response.text
            print(f"Gemini reply: {reply}")
            return jsonify({'reply': reply})
        else:
            msg = '죄송합니다. 현재 AI 답변 서비스를 사용할 수 없습니다.'
            if not os.getenv('GOOGLE_API_KEY'):
                msg += ' (API Key 미설정)'
            if context:
                 msg += f'\n[검색 결과 참고]\n{context}'
            return jsonify({'reply': msg}), 200

    except Exception as e:
        print(f"Error processing chat: {e}")
        return jsonify({'reply': '죄송합니다. 현재 서비스에 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7879, debug=True)
