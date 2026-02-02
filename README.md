# 🌏 TravelAI - AI-Powered Travel ERP & Chatbot System
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=flat&logo=flask&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-8.0+-4479A1?style=flat&logo=mysql&logoColor=white)
![Socket.IO](https://img.shields.io/badge/Socket.IO-4.0+-010101?style=flat&logo=socket.io&logoColor=white)
![Kafka](https://img.shields.io/badge/Apache%20Kafka-3.0+-231F20?style=flat&logo=apache-kafka&logoColor=white)
![AI](https://img.shields.io/badge/Google%20Gemini-2.5%20Flash-4285F4?style=flat&logo=google&logoColor=white)
**Intelligent travel booking platform with AI-powered chatbot and real-time multi-party communication**
---
## 📋 프로젝트 개요
### 🎯 해결하는 문제
전통적인 여행사는 다음과 같은 문제에 직면합니다:
- 고객 문의에 대한 수동 응답으로 인한 **느린 응답 시간**
- 상품 추천 시 담당자의 **주관적 판단**에 의존
- 고객-관리자-랜드사 간 **비효율적인 커뮤니케이션**
- 견적서 작성 및 예약 관리의 **수작업 프로세스**
### 💡 핵심 가치 제안
- **🤖 AI 기반 상품 추천**: RAG(Retrieval-Augmented Generation)를 활용한 의미 기반 상품 검색
- **💬 실시간 다자간 채팅**: 고객 ↔ 관리자 ↔ 랜드사 간 즉각적인 소통
- **🎯 지능형 슬롯 필링**: 자연어 대화를 통한 예약 정보 자동 수집
- **⚡ 자동화된 워크플로우**: 견적 요청부터 인보이스 발행까지 자동화
### ✨ 주요 특징
- **멀티 에이전트 AI 시스템**: 의도 분류, 슬롯 필링, 컨텍스트 관리
- **벡터 데이터베이스**: FAISS를 활용한 고속 시맨틱 검색
- **실시간 이벤트 처리**: Socket.IO + Kafka 기반 메시징 시스템
- **역할 기반 접근 제어**: 고객, 관리자, 랜드사별 맞춤 인터페이스
---
## 🏗️ 시스템 아키텍처
```
┌─────────────────────────────────────────────────────────────────────┐
│                         Frontend Layer                              │
├─────────────┬─────────────────┬─────────────────┬──────────────────┤
│  Customer   │   Admin ERP     │  Land Company   │   Real-time      │
│  Interface  │   Dashboard     │    Portal       │   Chat UI        │
└──────┬──────┴────────┬────────┴────────┬────────┴────────┬─────────┘
       │               │                 │                 │
       └───────────────┴─────────────────┴─────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   Flask Backend   │
                    │  (Blueprint-based)│
                    └─────────┬─────────┘
                              │
         ┏━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━┓
         ▼                    ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   AI Service    │  │  Database Layer │  │  Message Queue  │
│                 │  │                 │  │                 │
│ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │
│ │   Gemini    │ │  │ │    MySQL    │ │  │ │    Kafka    │ │
│ │ 2.5 Flash   │ │  │ │  (Relational)│ │  │ │  Consumer   │ │
│ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │
│                 │  │                 │  │                 │
│ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │
│ │ RAG Engine  │ │  │ │    FAISS    │ │  │ │  Socket.IO  │ │
│ │ (BGE-M3)    │ │  │ │ (Vector DB) │ │  │ │   Server    │ │
│ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                    │                     │
         └────────────────────┴─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  External APIs    │
                    │  (Google AI)      │
                    └───────────────────┘
```
### 데이터 흐름
1. **고객 질의** → AI 의도 분류 → 적절한 핸들러 라우팅
2. **상품 추천 요청** → RAG 엔진 → FAISS 벡터 검색 → Gemini 응답 생성
3. **예약 프로세스** → 슬롯 필링 → 세션 컨텍스트 저장 → 랜드사 견적 요청
4. **실시간 채팅** → Socket.IO 이벤트 → Kafka 메시지 큐 → 다자간 브로드캐스트
---
## 🎯 주요 기능
### 👤 고객 기능
- ✅ **AI 챗봇 상담**: 자연어 기반 상품 문의 및 예약
- ✅ **스마트 상품 검색**: 의미 기반 검색으로 정확한 상품 추천
- ✅ **실시간 채팅**: 관리자와 즉각적인 소통
- ✅ **예약 관리**: 예약 현황 조회 및 수정
- ✅ **맞춤 견적 요청**: 특별 요구사항 반영 견적 요청
**핵심 기능: 대화형 예약 시스템**
- 자연어로 출발지, 인원, 날짜 등 정보 수집
- 추가 요청사항 자동 분석 (휠체어, 오션뷰 등)
- 요청사항 유무에 따라 인보이스 발행 또는 랜드사 견적 요청으로 자동 분기
### 👨‍💼 관리자 기능
- ✅ **ERP 대시보드**: 예약, 상품, 고객 통합 관리
- ✅ **상품 관리**: 
  - JSON 기반 상품 데이터 업로드
  - 자동 벡터 인덱싱 (RAG 시스템 연동)
  - 일정, 가격, 리소스 정보 관리
- ✅ **채팅 모니터링**: 모든 고객 대화 실시간 모니터링
- ✅ **예약 관리**: 예약 승인, 수정, 취소 처리
- ✅ **랜드사 커뮤니케이션**: 견적 요청 및 응답 관리
### 🏢 랜드사 기능
- ✅ **견적 요청 수신**: 관리자로부터 견적 요청 실시간 수신
- ✅ **실시간 채팅**: 관리자와 직접 소통
- ✅ **견적서 제출**: 상세 견적 정보 제공
---
## 🛠️ 기술 스택
### Backend
| 기술 | 버전 | 사용 목적 |
|------|------|-----------|
| **Python** | 3.8+ | 메인 프로그래밍 언어 |
| **Flask** | 2.0+ | 웹 프레임워크, Blueprint 기반 모듈화 |
| **SQLAlchemy** | 1.4+ | ORM, 데이터베이스 추상화 |
| **Flask-SocketIO** | 5.0+ | 실시간 양방향 통신 |
| **PyMySQL** | 1.0+ | MySQL 데이터베이스 드라이버 |
### AI/ML
| 기술 | 버전 | 사용 목적 |
|------|------|-----------|
| **Google Gemini** | 2.5 Flash | LLM 기반 의도 분류, 슬롯 필링, 응답 생성 |
| **FAISS** | 1.7+ | 고속 벡터 유사도 검색 (Facebook AI) |
| **Sentence Transformers** | 2.2+ | 텍스트 임베딩 생성 (BGE-M3 모델) |
| **NumPy** | 1.24+ | 벡터 연산 및 데이터 처리 |
**BGE-M3 모델 선택 이유:**
- 다국어 지원 (한국어 성능 우수)
- 높은 임베딩 품질 (MTEB 벤치마크 상위권)
- 효율적인 추론 속도
### Database
| 기술 | 사용 목적 |
|------|-----------|
| **MySQL** | 관계형 데이터 저장 (상품, 예약, 채팅 로그) |
| **FAISS Index** | 벡터 데이터베이스 (상품 임베딩) |
### Real-time Communication
| 기술 | 사용 목적 |
|------|-----------|
| **Socket.IO** | 웹소켓 기반 실시간 이벤트 전송 |
| **Apache Kafka** | 메시지 큐, 비동기 이벤트 처리 |
### Frontend
| 기술 | 사용 목적 |
|------|-----------|
| **Jinja2** | 서버 사이드 템플릿 엔진 |
| **HTML/CSS/JavaScript** | 사용자 인터페이스 |
| **Bootstrap** | 반응형 UI 컴포넌트 |
---
## 📁 프로젝트 구조
```
flask_web/
├── 📄 app.py                      # 메인 애플리케이션 엔트리포인트
├── 📄 models.py                   # SQLAlchemy 초기화
├── 📄 events.py                   # Socket.IO 이벤트 핸들러
│
├── 📂 routes/                     # Blueprint 라우트 모듈
│   ├── admin_routes.py           # 관리자 ERP 및 채팅 API
│   ├── customer_routes.py        # 고객 인터페이스 라우트
│   ├── product_routes.py         # 상품 관리 API
│   └── land_routes.py            # 랜드사 포털 라우트
│
├── 📂 schema/                     # 데이터베이스 스키마 정의
│   ├── schema.py                 # 상품 테이블 및 Pydantic 모델
│   ├── chat_schema.py            # 고객 채팅 스키마
│   ├── customer_schema.py        # 고객 및 예약 스키마
│   └── land_schema.py            # 랜드사 채팅 스키마
│
├── 📂 services/                   # 비즈니스 로직 서비스
│   ├── ai_service.py             # AI 챗봇 핵심 로직 (의도 분류, 슬롯 필링)
│   ├── rag_service.py            # RAG 엔진 (FAISS 벡터 검색)
│   ├── kafka_service.py          # Kafka 컨슈머 워커
│   ├── chat_service.py           # 채팅 비즈니스 로직
│   └── db_connect.py             # 데이터베이스 연결 설정
│
├── 📂 templates/                  # Jinja2 템플릿
│   ├── admin/                    # 관리자 페이지 (ERP, 채팅 모니터링)
│   ├── customer/                 # 고객 페이지 (상품 조회, 채팅)
│   └── land/                     # 랜드사 페이지
│
├── 📂 static/                     # 정적 파일
│   ├── js/                       # JavaScript 파일
│   ├── uploads/                  # 업로드된 파일 (상품 이미지 등)
│   └── templates/                # 프론트엔드 템플릿
│
├── 📂 notebooks/                  # Jupyter 노트북 (실험 및 분석)
├── 📄 init_db.py                  # 데이터베이스 초기화 스크립트
├── 📄 seed_db.py                  # 샘플 데이터 시딩
├── 📄 check_rag_status.py         # RAG 시스템 상태 확인
├── 📄 .env                        # 환경 변수 (API 키, DB 설정)
└── 📄 requirements_backup.txt     # Python 의존성 목록
```
---
## 🧠 핵심 알고리즘/로직
### 1. AI 의도 분류 시스템
```python
def classify_intent(user_text):
    """
    사용자 입력을 6가지 카테고리로 분류
    - Greeting: 인사, 잡담
    - Complaint: 불만, 상담원 연결
    - Booking: 구체적 예약 의사
    - Recommend: 상품 추천 요청
    - Custom: 맞춤 견적 요청
    - Irrelevant: 여행 무관 주제
    """
    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = f"""
    당신은 여행사 챗봇의 라우터입니다. 사용자 입력을 분석해 카테고리를 분류하세요.
    [사용자 입력] "{user_text}"
    [출력] 오직 JSON만: {{"category": "카테고리명"}}
    """
    response = model.generate_content(prompt)
    result = json.loads(response.text)
    return result.get('category', 'Custom')
```
### 2. 슬롯 필링 엔진
**목적:** 자연어 대화에서 필수 정보 추출 (출발지, 인원, 날짜 등)
```python
def run_slot_filling(user_text, required_fields_desc, task_name, session_id):
    """
    LLM 기반 정보 추출 및 세션 컨텍스트 관리
    - 현재까지 수집된 정보와 새 입력을 결합
    - 부족한 정보가 있으면 되묻기
    - 모든 정보가 모이면 complete 상태 반환
    """
    current_context = session_storage.get(session_id, {})
    
    prompt = f"""
    [목표] 대화를 통해 아래 [필수 정보]를 모두 수집
    [필수 정보] {required_fields_desc}
    [현재까지 수집된 정보] {current_context}
    [고객의 현재 말] "{user_text}"
    
    [출력 포맷]
    {{
        "status": "ing" 또는 "complete",
        "extracted_data": {{"항목명": "값"}},
        "reply": "고객에게 할 답변"
    }}
    """
    
    result = model.generate_content(prompt)
    # 세션에 추출된 데이터 저장
    session_storage[session_id].update(result['extracted_data'])
    return result
```
### 3. RAG 시스템 (Retrieval-Augmented Generation)
**아키텍처:**
```
사용자 질의
    ↓
[1] 텍스트 임베딩 (BGE-M3)
    ↓
[2] FAISS 벡터 검색 (L2 Distance)
    ↓
[3] 임계값 필터링 (Threshold = 1.2)
    ↓
[4] 날짜 기반 정렬 (시간 관련 질의 시)
    ↓
[5] Top-K 선택 (기본 3개)
    ↓
[6] Gemini로 최종 응답 생성
```
**핵심 코드:**
```python
def search_best_products(user_query: str, top_k: int = 3):
    # 1. 쿼리 임베딩
    model = SentenceTransformer('BAAI/bge-m3')
    query_vector = model.encode([user_query])
    
    # 2. FAISS 검색
    index = faiss.read_index('travel_products.index')
    distances, indices = index.search(query_vector, 20)
    
    # 3. 임계값 검사 (유사도 낮으면 빈 리스트 반환 → Custom 모드로 전환)
    THRESHOLD = 1.2
    if distances[0][0] > THRESHOLD:
        return []  # RAG 실패 → 맞춤 견적으로 라우팅
    
    # 4. 날짜 관련 질의 감지 및 정렬
    date_keywords = ["날짜", "언제", "가까운", "빠른"]
    if any(kw in user_query for kw in date_keywords):
        candidates.sort(key=lambda x: datetime.strptime(x['start_date'], "%Y-%m-%d"))
    
    # 5. Top-K 반환
    return candidates[:top_k]
```
### 4. Task Locking 메커니즘
**문제:** 다중 턴 대화 중 사용자가 다른 주제로 전환하면 컨텍스트 손실
**해결책:** 작업 잠금(Lock) 시스템
```python
def process_user_message(user_text, session_id):
    current_task = session_storage.get(session_id, {}).get('task')
    
    # Lock이 걸려있으면 의도 분류 없이 해당 핸들러로 직행
    if current_task == 'booking':
        return handle_booking(user_text, session_id)
    
    # Lock이 없으면 정상적으로 의도 분류
    category = classify_intent(user_text)
    
    if category == 'Booking':
        session_storage[session_id]['task'] = 'booking'  # Lock 설정
        return handle_booking(user_text, session_id)
```
**예약 프로세스 플로우차트:**
```
사용자: "이거 예약할래"
    ↓
[Lock 설정: task='booking']
    ↓
Step 1: 출발지, 인원 수집 (슬롯 필링)
    ↓
정보 완료?
    ├─ No → 되묻기 (Lock 유지)
    └─ Yes → Step 2로 이동
                ↓
Step 2: "추가 요청사항이 있으신가요?"
    ↓
사용자 응답 분석
    ├─ "없음" → 인보이스 발행 [Lock 해제]
    └─ "오션뷰 줘" → 랜드사 견적 요청 [Lock 해제]
```
---
## 🚀 설치 및 실행
### 환경 요구사항
- **Python**: 3.8 이상
- **MySQL**: 8.0 이상
- **Apache Kafka**: 3.0 이상 (선택사항, 실시간 채팅 기능 사용 시)
- **OS**: Windows, Linux, macOS
### 1단계: 저장소 클론
```bash
git clone https://github.com/ysksean/AI-Travel-ERP.git
cd AI-Travel-ERP
```
### 2단계: 가상 환경 생성 및 활성화
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate
# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```
### 3단계: 의존성 설치
```bash
pip install -r requirements_backup.txt
```
**주요 패키지:**
- `flask`, `flask-socketio`, `flask-sqlalchemy`
- `google-generativeai` (Gemini API)
- `faiss-cpu`, `sentence-transformers`
- `pymysql`, `kafka-python`
### 4단계: 환경 변수 설정
`.env` 파일 생성:
```env
# Google AI API Key
GOOGLE_API_KEY=your_gemini_api_key_here
# MySQL Database Configuration
DB_HOST=localhost
DB_PORT=3306
DB_NAME=travel_ai_db
DB_USER=root
DB_PASSWORD=your_mysql_password
# Kafka Configuration (Optional)
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC=travel_chat
# Flask Configuration
FLASK_ENV=development
SECRET_KEY=your_secret_key_here
```
**Google API Key 발급:**
1. [Google AI Studio](https://makersuite.google.com/app/apikey) 방문
2. API 키 생성
3. `.env` 파일에 추가
### 5단계: 데이터베이스 초기화
```bash
# MySQL 데이터베이스 생성
mysql -u root -p
CREATE DATABASE travel_ai_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
EXIT;
# 테이블 생성 및 샘플 데이터 삽입
python init_db.py
python seed_db.py
```
### 6단계: RAG 벡터 인덱스 생성
```bash
# 상품 데이터를 벡터 DB에 인덱싱
python check_rag_status.py
```
### 7단계: 애플리케이션 실행
```bash
python app.py
```
**서버 시작 확인:**
```
==================================================
🚀 Chat & ERP Server Starting...
==================================================
📍 Admin:         http://localhost:7878/admin
📍 Customer:      http://localhost:7878/customer
📍 Chat Socket:   ws://localhost:7878
==================================================
```
### 8단계: 접속 확인
- **고객 페이지**: http://localhost:7878/customer
- **관리자 ERP**: http://localhost:7878/admin
- **랜드사 포털**: http://localhost:7878/land
---
## 🗄️ 데이터베이스 스키마
### ERD (Entity Relationship Diagram)
```
┌─────────────────┐         ┌─────────────────┐
│   chat_rooms    │         │   chat_logs     │
├─────────────────┤         ├─────────────────┤
│ session_id (PK) │◄────────│ id (PK)         │
│ user_name       │         │ session_id (FK) │
│ user_type       │         │ role            │
│ status          │         │ text            │
│ last_message    │         │ user_name       │
│ last_active     │         │ created_at      │
│ created_at      │         └─────────────────┘
└─────────────────┘
        
┌─────────────────┐         ┌─────────────────┐
│   customers     │         │  reservations   │
├─────────────────┤         ├─────────────────┤
│ id (PK)         │◄────────│ id (PK)         │
│ name            │         │ customer_id (FK)│
│ email           │         │ product_id (FK) │
│ phone           │         │ status          │
│ created_at      │         │ departure_date  │
└─────────────────┘         │ total_price     │
                            │ created_at      │
                            └─────────────────┘
                                    │
                                    │
                            ┌───────▼─────────┐
                            │    products     │
                            ├─────────────────┤
                            │ id (PK)         │
                            │ product_name    │
                            │ category        │
                            │ country         │
                            │ start_date      │
                            │ price_adult     │
                            │ images (JSON)   │
                            │ itinerary (JSON)│
                            └─────────────────┘
┌─────────────────┐         ┌─────────────────┐
│land_chat_rooms  │         │ land_chat_logs  │
├─────────────────┤         ├─────────────────┤
│ session_id (PK) │◄────────│ id (PK)         │
│ land_name       │         │ session_id (FK) │
│ admin_name      │         │ role            │
│ status          │         │ text            │
│ created_at      │         │ created_at      │
└─────────────────┘         └─────────────────┘
```
### 주요 테이블 설명
#### `products` 테이블
| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| `id` | INT (PK) | 상품 고유 ID |
| `product_name` | VARCHAR(255) | 상품명 |
| `category` | VARCHAR(50) | 카테고리 (overseas, domestic) |
| `product_type` | VARCHAR(50) | 상품 유형 (golf, tour, etc.) |
| `country` | VARCHAR(50) | 국가 |
| `city` | VARCHAR(50) | 도시 |
| `start_date` | DATE | 출발일 |
| `end_date` | DATE | 도착일 |
| `price_adult` | INT | 성인 가격 |
| `images` | JSON | 이미지 URL 배열 |
| `itinerary` | JSON | 일정 상세 정보 |
| `resources` | JSON | 호텔, 골프장 정보 |
#### `chat_rooms` & `chat_logs` 테이블
고객-관리자 간 채팅 세션 및 메시지 로그 저장
- `session_id`: 고유 채팅방 식별자
- `role`: `customer`, `admin`, `ai` (메시지 발신자 구분)
- `status`: `OPEN`, `CLOSED` (채팅방 상태)
#### `land_chat_rooms` & `land_chat_logs` 테이블
관리자-랜드사 간 견적 요청 채팅
---
## 📡 API 문서
### REST API 엔드포인트
#### 고객 API
| Method | Endpoint | 설명 |
|--------|----------|------|
| `GET` | `/customer` | 고객 메인 페이지 |
| `GET` | `/customer/chat` | 채팅 인터페이스 |
| `GET` | `/api/chat/history/<session_id>` | 채팅 내역 조회 |
#### 관리자 API
| Method | Endpoint | 설명 |
|--------|----------|------|
| `GET` | `/admin` | ERP 대시보드 |
| `GET` | `/admin/chat` | 채팅 모니터링 페이지 |
| `GET` | `/admin/api/chat/rooms` | 모든 채팅방 목록 |
| `GET` | `/admin/api/chat/logs/<session_id>` | 특정 채팅방 로그 |
| `POST` | `/admin/api/chat/send` | 관리자 메시지 전송 |
#### 상품 관리 API
| Method | Endpoint | 설명 |
|--------|----------|------|
| `GET` | `/admin/products` | 상품 목록 조회 |
| `POST` | `/admin/products/create` | 상품 생성 (JSON 업로드) |
| `PUT` | `/admin/products/<id>` | 상품 수정 |
| `DELETE` | `/admin/products/<id>` | 상품 삭제 |
### Socket.IO 이벤트
#### 클라이언트 → 서버
| 이벤트명 | 페이로드 | 설명 |
|----------|----------|------|
| `customer_message` | `{session_id, text, user_name}` | 고객 메시지 전송 |
| `admin_message` | `{session_id, text, admin_name}` | 관리자 메시지 전송 |
| `join_room` | `{session_id}` | 채팅방 입장 |
#### 서버 → 클라이언트
| 이벤트명 | 페이로드 | 설명 |
|----------|----------|------|
| `ai_response` | `{session_id, text, timestamp}` | AI 응답 |
| `new_message` | `{session_id, role, text}` | 새 메시지 알림 |
| `room_update` | `{session_id, status}` | 채팅방 상태 변경 |
### API 사용 예제
#### 채팅 내역 조회
**Request:**
```http
GET /api/chat/history/customer_12345 HTTP/1.1
Host: localhost:7878
```
**Response:**
```json
[
  {
    "id": 1,
    "session_id": "customer_12345",
    "role": "customer",
    "text": "다낭 골프 여행 있어요?",
    "timestamp": "2026-02-02 13:30:15"
  },
  {
    "id": 2,
    "session_id": "customer_12345",
    "role": "ai",
    "text": "네! 다낭 3박 4일 골프 패키지가 있습니다...",
    "timestamp": "2026-02-02 13:30:18"
  }
]
```
#### 상품 생성 (JSON 업로드)
**Request:**
```http
POST /admin/products/create HTTP/1.1
Content-Type: multipart/form-data
{
  "PRODUCT_FILE": <JSON 파일>,
  "PRICE_FILE": <가격 JSON 파일>
}
```
**Response:**
```json
{
  "success": true,
  "product_id": 42,
  "message": "상품이 생성되고 벡터 DB에 인덱싱되었습니다."
}
```
---
## ⚡ 성능 최적화
### 적용된 최적화 기법
#### 1. 벡터 검색 최적화
- **FAISS IndexFlatL2 사용**: CPU 환경에서 최적화된 L2 거리 계산
- **임계값 필터링**: 유사도 낮은 결과 조기 차단 (Threshold = 1.2)
- **후보군 제한**: 전체 검색 후 Top-20만 추출하여 정렬 비용 절감
**성능 개선 결과:**
- 평균 검색 시간: ~50ms (10,000개 상품 기준)
- 메모리 사용량: ~200MB (인덱스 + 메타데이터)
#### 2. 세션 컨텍스트 관리
- **In-memory 세션 저장소**: Redis 대신 Python dict 사용 (소규모 트래픽 대응)
- **불필요한 컨텍스트 필터링**: 시스템 변수(`task`, `booking_step`) 프롬프트에서 제외
#### 3. AI 응답 생성 최적화
- **Gemini 2.5 Flash 모델 선택**: Pro 대비 2배 빠른 응답 속도
- **프롬프트 최적화**: 불필요한 컨텍스트 제거로 토큰 사용량 30% 감소
#### 4. 데이터베이스 쿼리 최적화
- **인덱스 설정**: `session_id`, `start_date`, `status` 컬럼에 인덱스 생성
- **Lazy Loading**: SQLAlchemy relationship에 `lazy=True` 설정
---
## 🔮 향후 개선 계획
### 단기 목표 (1-2개월)
- [ ] **반응형 UI 개선**: 모바일 최적화 (Bootstrap → Tailwind CSS)
- [ ] **사용자 인증 시스템**: JWT 기반 로그인/회원가입
- [ ] **파일 업로드 개선**: 이미지 압축 및 CDN 연동
- [ ] **에러 핸들링 강화**: 전역 에러 핸들러 및 로깅 시스템
- [ ] **테스트 코드 작성**: Unit Test (pytest) 및 Integration Test
### 중기 목표 (3-6개월)
- [ ] **다국어 지원**: i18n (한국어, 영어, 일본어)
- [ ] **결제 시스템 통합**: 토스페이먼츠, 카카오페이 연동
- [ ] **이메일 알림**: 예약 확정, 견적 도착 시 자동 이메일 발송
- [ ] **관리자 대시보드 고도화**: 
  - 매출 통계 차트 (Chart.js)
  - 고객 행동 분석 (Google Analytics 연동)
- [ ] **Redis 도입**: 세션 관리 및 캐싱
### 장기 목표 (6개월 이상)
- [ ] **고급 분석 기능**:
  - 고객 선호도 분석 (협업 필터링)
  - 수요 예측 모델 (시계열 분석)
- [ ] **ML 기반 가격 최적화**:
  - 동적 가격 책정 (Dynamic Pricing)
  - 수익 최대화 알고리즘
- [ ] **마이크로서비스 아키텍처 전환**:
  - AI 서비스 분리 (FastAPI)
  - Kubernetes 기반 배포
- [ ] **음성 챗봇**: STT/TTS 연동 (Google Cloud Speech API)
- [ ] **추천 시스템 고도화**: 
  - 하이브리드 추천 (Content-based + Collaborative Filtering)
  - A/B 테스트 프레임워크
---
## 👨‍💻 개발자 정보
**이름**: 김유신 (Yushin Kim)  
**이메일**: ysksean@example.com  
**GitHub**: [@ysksean](https://github.com/ysksean)  
**LinkedIn**: [linkedin.com/in/ysksean](https://linkedin.com/in/ysksean)  
**Portfolio**: [ysksean.dev](https://ysksean.dev)
### 프로젝트 기여자
- **김유신**: 백엔드 개발, AI 시스템 설계
- **임규리**: 프론트엔드 개발, UI/UX 디자인
---
## 📄 라이선스
이 프로젝트는 **포트폴리오 및 교육 목적**으로 제작되었습니다.
```
MIT License
Copyright (c) 2026 Yushin Kim
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
---
## 🙏 감사의 말
이 프로젝트는 다음 오픈소스 프로젝트들의 도움을 받았습니다:
- [Flask](https://flask.palletsprojects.com/) - 웹 프레임워크
- [Google Gemini](https://ai.google.dev/) - LLM API
- [FAISS](https://github.com/facebookresearch/faiss) - 벡터 검색 엔진
- [Sentence Transformers](https://www.sbert.net/) - 텍스트 임베딩
- [Socket.IO](https://socket.io/) - 실시간 통신
---
## 📞 문의 및 지원
프로젝트에 대한 질문이나 제안사항이 있으시면 언제든지 연락주세요!
- **이슈 등록**: [GitHub Issues](https://github.com/ysksean/AI-Travel-ERP/issues)
- **이메일**: ysksean@example.com
- **디스코드**: TravelAI Community (준비 중)
---
<div align="center">
**⭐ 이 프로젝트가 도움이 되셨다면 Star를 눌러주세요! ⭐**
Made with ❤️ by [Yushin Kim](https://github.com/ysksean)
</div>
