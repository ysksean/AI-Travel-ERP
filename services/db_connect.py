# services/db_connect.py
# MySQL 데이터베이스 연결 설정
# flask_web과 동일한 데이터베이스 사용

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

# MySQL 연결 URL
# flask_web과 동일한 데이터베이스 연결 설정
DB_PASSWORD = os.getenv("DB_PASSWORD", "0000")  # 환경변수 또는 직접 입력
DB_URL = f"mysql+pymysql://root:{DB_PASSWORD}@localhost:3306/travel_erp?charset=utf8mb4"

# SQLAlchemy 엔진 생성
# charset=utf8mb4는 한글 저장을 위해 필수입니다
engine = create_engine(
    DB_URL,
    pool_pre_ping=True,  # 연결 유효성 검사
    pool_recycle=3600,   # 1시간마다 연결 재생성
    echo=False  # SQL 쿼리 로그 출력 (디버깅 시 True로 변경)
)

# 세션 팩토리 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base 클래스 (모델 상속용)
Base = declarative_base()


# 데이터베이스 세션 의존성 (FastAPI 스타일, Flask에서도 사용 가능)
def get_db():
    """
    데이터베이스 세션 생성 및 반환
    사용 예시:
        db = next(get_db())
        try:
            # DB 작업 수행
            ...
        finally:
            db.close()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

