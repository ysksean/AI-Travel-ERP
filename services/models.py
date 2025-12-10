# services/models.py
# MySQL 데이터베이스 모델 정의
# flask_web과 동일한 모델 사용

from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey, Numeric, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from services.db_connect import Base


class Product(Base):
    """상품 테이블"""
    __tablename__ = 'products'

    id = Column(Integer, primary_key=True, autoincrement=True)

    # 기본 정보
    product_name = Column(String(255), nullable=False)
    product_type = Column(String(50))  # 'domestic' or 'overseas'

    # 위치 정보
    country = Column(String(100))
    city = Column(String(100))
    departure_port = Column(String(100))

    # 일정 정보는 ProductPrice 테이블에서만 관리
    # (하나의 상품이 여러 기간 옵션을 가질 수 있음: 2박3일, 3박4일, 4박5일 등)

    # 상세 정보 (JSON 형식으로 저장)
    details_json = Column(JSON)  # inclusions, exclusions, special_notes 등

    # 호텔 정보 (JSON 배열)
    hotels_json = Column(JSON)  # 호텔 리스트

    # 골프장 정보 (JSON 배열)
    golf_courses_json = Column(JSON)  # 골프장 리스트

    # 일정 옵션 (JSON 배열)
    itinerary_options_json = Column(JSON)  # 기간별 일정 옵션

    # 항공 정보
    flight_info_json = Column(JSON)  # 항공편 정보

    # AI 생성 콘텐츠
    ai_content_json = Column(JSON)  # AI 생성 본문

    # 상태
    status = Column(String(50), default='draft')  # 'draft' or 'published'

    # 타임스탬프
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 관계
    prices = relationship("ProductPrice", back_populates="product", cascade="all, delete-orphan")


class ProductPrice(Base):
    """상품 가격 테이블"""
    __tablename__ = 'product_prices'

    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(Integer, ForeignKey('products.id', ondelete='CASCADE'), nullable=False)

    departure_date = Column(DateTime, nullable=False)
    night_count = Column(Integer)
    day_count = Column(Integer)
    group_size = Column(Integer, default=1)
    price_adult = Column(Numeric(10, 0))  # 성인 가격
    price_child = Column(Numeric(10, 0), nullable=True)  # 아동 가격
    status = Column(String(50), default='available')  # 'available', 'sold_out', etc.

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 관계
    product = relationship("Product", back_populates="prices")

