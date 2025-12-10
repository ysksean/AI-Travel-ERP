import sys
import os

try:
    import google.generativeai as genai
except ImportError:
    genai = None
    print("Warning: google-generativeai module not found.")

from flask import Flask, render_template, request, jsonify
from services.db_connect import SessionLocal
from services.models import Product, ProductPrice
from sqlalchemy import desc, or_, func, and_
import re

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
    """
    메인 페이지 - DB에서 published 상태의 상품을 조회하여 표시
    """
    try:
        db = SessionLocal()
        try:
            # published 상태인 상품만 조회 (고객용 사이트는 읽기 전용)
            products = db.query(Product).filter(
                Product.status == 'published'
            ).order_by(desc(Product.created_at)).limit(20).all()
            
            # 상품 데이터를 템플릿에 맞는 형식으로 변환
            products_data = []
            for product in products:
                # 최저 가격 찾기
                min_price = None
                if product.prices:
                    available_prices = [p.price_adult for p in product.prices if p.price_adult is not None]
                    if available_prices:
                        min_price = min(available_prices)
                
                # 이미지 URL 추출 (details_json 또는 ai_content_json에서)
                image_url = "https://images.unsplash.com/photo-1540206351-d6465b3ac5c1?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80"  # 기본 이미지
                if product.details_json and isinstance(product.details_json, dict):
                    if 'images' in product.details_json and product.details_json['images']:
                        image_url = product.details_json['images'][0] if isinstance(product.details_json['images'], list) else image_url
                
                product_dict = {
                    'id': product.id,
                    'title': product.product_name,
                    'price': f"{int(min_price):,}" if min_price else "문의",
                    'original_price': None,  # 필요시 추가
                    'discount': None,  # 필요시 추가
                    'image': image_url,
                    'tags': [f"#{product.country}", f"#{product.city}"] if product.country else [],
                    'country': product.country,
                    'city': product.city,
                    'nights': product.nights,
                    'days': product.days
                }
                products_data.append(product_dict)
            
            # mock_data 구조 유지하면서 실제 상품 데이터 추가
            data = mock_data.copy()
            if products_data:
                # 실제 상품 데이터로 교체
                data['products_a'] = products_data[:4]  # 상단 4개
                data['products_b'] = products_data[4:8] if len(products_data) > 4 else products_data[4:]  # 하단 4개
            
            return render_template('index.html', data=data)
        finally:
            db.close()
    except Exception as e:
        print(f"⚠️  Error loading products from database: {e}")
        print("   Falling back to mock data")
        # DB 연결 실패 시 mock_data 사용
        return render_template('index.html', data=mock_data)

@app.route('/products/<int:product_id>')
def product_detail(product_id):
    """
    상품 상세 페이지 - DB에서 상품 정보 조회
    """
    try:
        db = SessionLocal()
        try:
            product = db.query(Product).filter(Product.id == product_id).first()
            
            if not product:
                return render_template('404.html', message="상품을 찾을 수 없습니다."), 404
            
            # published 상태가 아니면 404
            if product.status != 'published':
                return render_template('404.html', message="상품을 찾을 수 없습니다."), 404
            
            # 가격 정보 가져오기
            prices = db.query(ProductPrice).filter(
                ProductPrice.product_id == product_id,
                ProductPrice.status == 'available'
            ).order_by(ProductPrice.departure_date).all()
            
            return render_template('product_detail.html', product=product, prices=prices)
        finally:
            db.close()
    except Exception as e:
        print(f"⚠️  Error loading product detail: {e}")
        return render_template('404.html', message="상품 정보를 불러올 수 없습니다."), 500

def parse_query_intent(query_text):
    """
    자연어 질문에서 검색 조건을 추출
    Returns: dict with 'nights', 'days', 'max_price', 'location_keywords', 'find_cheapest'
    """
    intent = {
        'nights': None,
        'days': None,
        'max_price': None,
        'location_keywords': [],
        'find_cheapest': False  # "가장 싼", "최저가" 같은 요청 감지
    }
    
    # "가장 싼", "최저가", "제일 저렴한" 같은 키워드 감지
    cheapest_keywords = ['가장 싼', '가장 저렴한', '최저가', '제일 싼', '제일 저렴한', '싼', '저렴한']
    if any(keyword in query_text for keyword in cheapest_keywords):
        intent['find_cheapest'] = True
    
    # 1. 박/일 수 추출 (예: "3박 4일", "3박", "4일")
    # 패턴: 숫자 + "박" + (선택적) 숫자 + "일"
    night_day_pattern = r'(\d+)\s*박(?:\s*(\d+)\s*일)?'
    match = re.search(night_day_pattern, query_text)
    if match:
        intent['nights'] = int(match.group(1))
        if match.group(2):
            intent['days'] = int(match.group(2))
        else:
            # "3박"만 있으면 days는 nights+1로 추정
            intent['days'] = intent['nights'] + 1
    
    # 2. 가격 범위 추출 (예: "100만원 이하", "100만원 미만", "100만원")
    price_patterns = [
        (r'(\d+)\s*만\s*원\s*(?:이하|미만|이내)', lambda m: int(m.group(1)) * 10000),
        (r'(\d+)\s*만\s*원', lambda m: int(m.group(1)) * 10000),
        (r'(\d+)\s*원\s*(?:이하|미만|이내)', lambda m: int(m.group(1))),
    ]
    
    for pattern, converter in price_patterns:
        match = re.search(pattern, query_text)
        if match:
            intent['max_price'] = converter(match)
            break
    
    # 3. 위치 키워드 추출 (일반적인 여행지명)
    location_keywords = []
    common_locations = [
        '홋카이도', '도쿄', '오사카', '교토', '후쿠오카', '오키나와', '미야코지마',
        '제주', '제주도', '부산', '서울',
        '다낭', '호이안', '하노이', '호치민', '나트랑',
        '방콕', '푸켓', '치앙마이',
        '발리', '자카르타',
        '세부', '보라카이', '마닐라',
        '코타키나발루', '쿠알라룸푸르', '랑카위',
        '싱가포르', '홍콩', '마카오',
        '상하이', '베이징', '하이난',
        '시드니', '멜버른', '골드코스트',
        '두바이', '아부다비',
        '이스탄불', '카파도키아',
        '로마', '밀라노', '피렌체', '베네치아',
        '파리', '런던', '바르셀로나', '마드리드'
    ]
    
    for loc in common_locations:
        if loc in query_text:
            location_keywords.append(loc)
    
    # 일반 키워드도 추가 (명확한 위치가 아닌 경우)
    if not location_keywords:
        # 한글 단어 추출 (2글자 이상)
        korean_words = re.findall(r'[가-힣]{2,}', query_text)
        location_keywords = [w for w in korean_words if len(w) >= 2]
    
    intent['location_keywords'] = location_keywords
    
    return intent


def search_products_from_db(query_text):
    """
    DB에서 상품을 검색하여 텍스트 형식으로 반환
    자연어 파싱을 통해 정확한 필터링 수행
    "가장 싼 출발일" 요청 시 ProductPrice에서 직접 최저가 찾기
    """
    try:
        db = SessionLocal()
        try:
            # 1. 자연어 의도 파싱
            intent = parse_query_intent(query_text)
            print(f"🔍 Parsed Intent: {intent}")
            
            # 2. "가장 싼 출발일" 요청인 경우 특별 처리
            if intent['find_cheapest'] and (intent['nights'] is not None or intent['days'] is not None):
                return find_cheapest_departure_date(db, intent)
            
            # 3. 기본 쿼리: published 상태인 상품만
            query = db.query(Product).filter(Product.status == 'published')
            
            # 4. 박/일 수 필터링 (정확한 매칭)
            if intent['nights'] is not None:
                query = query.filter(Product.nights == intent['nights'])
            if intent['days'] is not None:
                query = query.filter(Product.days == intent['days'])
            
            # 5. 위치 필터링 (상품명, 국가, 도시에서 검색)
            location_conditions = []
            if intent['location_keywords']:
                for keyword in intent['location_keywords']:
                    keyword_like = f"%{keyword}%"
                    location_conditions.append(Product.product_name.like(keyword_like))
                    location_conditions.append(Product.country.like(keyword_like))
                    location_conditions.append(Product.city.like(keyword_like))
            
            # 위치 조건이 있으면 적용, 없으면 전체 검색
            if location_conditions:
                query = query.filter(or_(*location_conditions))
            
            # 6. 상품 조회 (최대 10개)
            products = query.limit(10).all()
            
            # 7. 가격 필터링 (ProductPrice와 조인하여 필터링)
            filtered_products = []
            for product in products:
                # ProductPrice에서 최저 가격 찾기
                if product.prices:
                    available_prices = [
                        (p, float(p.price_adult)) for p in product.prices 
                        if p.price_adult is not None and p.status == 'available'
                    ]
                    if available_prices:
                        # 최저가 출발일 찾기
                        cheapest_price_obj, min_price = min(available_prices, key=lambda x: x[1])
                        # 가격 필터 적용
                        if intent['max_price'] is None or min_price <= intent['max_price']:
                            filtered_products.append((product, min_price, cheapest_price_obj))
                else:
                    # 가격 정보가 없어도 포함 (가격 필터가 없으면)
                    if intent['max_price'] is None:
                        filtered_products.append((product, None, None))
            
            # 8. "가장 싼" 요청이면 가격순 정렬
            if intent['find_cheapest']:
                filtered_products.sort(key=lambda x: x[1] if x[1] is not None else float('inf'))
            
            # 최대 5개만 반환
            filtered_products = filtered_products[:5]
            
            if not filtered_products:
                return ""
            
            # 9. 상품 정보를 텍스트로 변환
            product_texts = []
            for product, min_price, cheapest_price_obj in filtered_products:
                product_info = f"상품명: {product.product_name}"
                if product.country:
                    product_info += f", 국가: {product.country}"
                if product.city:
                    product_info += f", 도시: {product.city}"
                if product.nights:
                    if product.days:
                        product_info += f", 기간: {product.nights}박 {product.days}일"
                    else:
                        product_info += f", 기간: {product.nights}박"
                
                # 가격 정보 (price_adult 사용)
                if min_price is not None:
                    # 최저가 출발일 정보 포함
                    if cheapest_price_obj and cheapest_price_obj.departure_date:
                        departure_str = cheapest_price_obj.departure_date.strftime('%Y년 %m월 %d일')
                        product_info += f", 최저가 출발일: {departure_str}, 가격: {int(min_price):,}원"
                    else:
                        product_info += f", 성인 가격: {int(min_price):,}원"
                    
                    # 모든 가격 옵션도 포함 (선택적)
                    price_options = []
                    for price in product.prices:
                        if price.price_adult is not None and price.status == 'available':
                            date_str = price.departure_date.strftime('%Y-%m-%d') if price.departure_date else ''
                            price_options.append(f"{date_str} {int(float(price.price_adult)):,}원")
                    
                    if len(price_options) > 1:
                        product_info += f" (출발일별 가격: {', '.join(price_options[:5])})"
                else:
                    product_info += ", 가격: 문의"
                
                # 상세 정보 추가
                if product.details_json and isinstance(product.details_json, dict):
                    if 'inclusions' in product.details_json and product.details_json['inclusions']:
                        inc = product.details_json['inclusions']
                        if isinstance(inc, list):
                            product_info += f", 포함사항: {', '.join(inc[:3])}"
                        elif isinstance(inc, str):
                            product_info += f", 포함사항: {inc[:100]}"
                
                product_texts.append(product_info)
            
            return "\n".join(product_texts)
        finally:
            db.close()
    except Exception as e:
        print(f"⚠️  Error searching products from DB: {e}")
        import traceback
        traceback.print_exc()
        return ""


def find_cheapest_departure_date(db, intent):
    """
    특정 기간(nights/days)의 상품 중 가장 싼 출발일을 찾기
    ProductPrice 테이블에서 직접 조회하여 최저가 찾기
    """
    try:
        # ProductPrice와 Product를 조인하여 조회
        query = db.query(ProductPrice, Product).join(
            Product, ProductPrice.product_id == Product.id
        ).filter(
            Product.status == 'published',
            ProductPrice.status == 'available',
            ProductPrice.price_adult.isnot(None)
        )
        
        # 박/일 수 필터링
        if intent['nights'] is not None:
            query = query.filter(Product.nights == intent['nights'])
        if intent['days'] is not None:
            query = query.filter(Product.days == intent['days'])
        
        # 위치 필터링
        if intent['location_keywords']:
            location_conditions = []
            for keyword in intent['location_keywords']:
                keyword_like = f"%{keyword}%"
                location_conditions.append(Product.product_name.like(keyword_like))
                location_conditions.append(Product.country.like(keyword_like))
                location_conditions.append(Product.city.like(keyword_like))
            if location_conditions:
                query = query.filter(or_(*location_conditions))
        
        # 가격 필터링
        if intent['max_price'] is not None:
            query = query.filter(ProductPrice.price_adult <= intent['max_price'])
        
        # 모든 결과 가져오기
        results = query.all()
        
        if not results:
            return ""
        
        # 가격순 정렬하여 최저가 찾기
        results.sort(key=lambda x: float(x[0].price_adult))
        
        # 최저가 상위 5개 반환
        product_texts = []
        for price_obj, product in results[:5]:
            price_value = float(price_obj.price_adult)
            departure_str = price_obj.departure_date.strftime('%Y년 %m월 %d일') if price_obj.departure_date else '날짜 미정'
            
            product_info = f"상품명: {product.product_name}"
            if product.country:
                product_info += f", 국가: {product.country}"
            if product.city:
                product_info += f", 도시: {product.city}"
            if product.nights:
                if product.days:
                    product_info += f", 기간: {product.nights}박 {product.days}일"
                else:
                    product_info += f", 기간: {product.nights}박"
            
            product_info += f", 출발일: {departure_str}, 가격: {int(price_value):,}원"
            
            # 그룹 사이즈 정보
            if price_obj.group_size:
                product_info += f" ({price_obj.group_size}인 기준)"
            
            product_texts.append(product_info)
        
        return "\n".join(product_texts)
    except Exception as e:
        print(f"⚠️  Error finding cheapest departure date: {e}")
        import traceback
        traceback.print_exc()
        return ""

@app.route('/chat', methods=['POST'])
def chat():
    print("Chat endpoint called")
    data = request.json
    user_message = data.get('message')
    print(f"User message: {user_message}")

    if not user_message:
        return jsonify({'reply': '메시지를 입력해주세요.'}), 400

    try:
        # 1. DB에서 상품 검색
        db_products = search_products_from_db(user_message)
        print(f"DB 검색 결과: {db_products[:200] if db_products else 'None'}")
        
        # 2. RAG Retrieval
        context = ""
        if rag_engine:
            print("Searching in RAG engine...")
            context = rag_engine.search(user_message)
            print(f"Retrieved context: {context}")
        else:
            print("RAG engine not available")
        
        # 3. DB 상품 정보와 RAG 컨텍스트 결합
        full_context = ""
        if db_products:
            full_context += f"[현재 보유 상품 정보]\n{db_products}\n\n"
        if context:
            full_context += f"[추가 정보]\n{context}"
        
        # 4. Gemini Generation
        if genai and os.getenv('GOOGLE_API_KEY'):
            model_name = os.getenv('MODEL_NAME', 'gemini-2.5-flash-lite')
            try:
                model = genai.GenerativeModel(model_name)
                print(f"Using model: {model_name}")
            except Exception:
                model = genai.GenerativeModel('gemini-pro')
                print("Fallback to model: gemini-pro")
            prompt = f"""
            You are a helpful travel agent for 'AI Hanatour'.
            Answer the user's question using the product information provided below.
            
            IMPORTANT RULES:
            1. If product information is provided in the Context, you MUST use it to answer the question.
            2. If the user asks for "가장 싼" (cheapest) or "최저가" (lowest price), find the product with the lowest price from the context.
            3. If the user asks about specific dates or departure dates, use the departure date information from the context.
            4. Always provide specific product names, prices, and departure dates when available in the context.
            5. If no product information is available, politely inform the user that you couldn't find matching products.
            
            Context (Product Information from Database):
            {full_context if full_context else "No product information found in database."}
            
            User Question:
            {user_message}
            
            Answer in Korean, providing specific details from the context:
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
            if db_products:
                msg += f'\n\n[검색된 상품 정보]\n{db_products}'
            if context:
                msg += f'\n\n[추가 정보]\n{context}'
            return jsonify({'reply': msg}), 200

    except Exception as e:
        print(f"Error processing chat: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'reply': '죄송합니다. 현재 서비스에 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7879, debug=True)
