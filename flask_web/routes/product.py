# routes/product.py

from flask import Blueprint, request, jsonify
import os
import uuid
import traceback
from datetime import datetime
from sqlalchemy import text
from services.parsing_service import UniversalTravelAI
from services.db_connect import SessionLocal
from services.models import Product, ProductPrice

bp = Blueprint('product', __name__, url_prefix='/product')

# 분석기 인스턴스 생성 (서버 시작 시 1회 생성)
travel_ai = UniversalTravelAI()

# 임시 파일 저장 경로
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)


@bp.route('/analyze', methods=['POST'])
def analyze_product():
    """
    프론트엔드에서 받은 파일을 임시 저장하고 AI 분석을 요청합니다.
    """
    # 1. 필수 파일 확인
    if 'product_file' not in request.files:
        return jsonify({"error": "No product_file provided"}), 400

    product_file = request.files['product_file']
    price_file = request.files.get('price_file')  # 없을 수도 있음

    if product_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 2. 고유 파일명 생성 (동시성 문제 해결)
    unique_id = str(uuid.uuid4())
    p_ext = os.path.splitext(product_file.filename)[1]
    temp_product_path = os.path.join(TEMP_DIR, f"{unique_id}_prod{p_ext}")

    temp_price_path = None

    try:
        # 3. 파일 저장
        product_file.save(temp_product_path)

        if price_file and price_file.filename != '':
            pr_ext = os.path.splitext(price_file.filename)[1]
            temp_price_path = os.path.join(TEMP_DIR, f"{unique_id}_price{pr_ext}")
            price_file.save(temp_price_path)

        # 4. 서비스 로직 호출
        # price_file이 없으면 None이 전달되고, parsing_service 내부에서 자동으로 처리됨
        result = travel_ai.analyze(temp_product_path, temp_price_path)

        return jsonify({
            "status": "success",
            "data": result
        })

    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        # 5. 임시 파일 삭제 (Cleanup)
        if os.path.exists(temp_product_path):
            os.remove(temp_product_path)
        if temp_price_path and os.path.exists(temp_price_path):
            os.remove(temp_price_path)


@bp.route('/save', methods=['POST'])
def save_product():
    """
    상품 저장 엔드포인트 (임시저장/게시)
    """
    db = None
    
    try:
        # DB 연결 테스트
        print("\n" + "="*80)
        print("🔌 Testing DB Connection...")
        try:
            db = SessionLocal()
            # 간단한 쿼리로 연결 테스트
            db.execute(text("SELECT 1"))
            print("   ✅ DB Connection OK")
        except Exception as conn_error:
            print(f"   ❌ DB Connection FAILED: {conn_error}")
            print(f"   Error Type: {type(conn_error).__name__}")
            traceback.print_exc()
            if db:
                db.close()
            return jsonify({
                "status": "error",
                "message": f"Database connection failed: {str(conn_error)}"
            }), 500
        
        print("="*80)
        # 1. 요청 수신 시 디버깅 로그
        print("🚀 Save Request Received")
        
        if not request.is_json:
            print("❌ ERROR: Request is not JSON format")
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        print(f"📦 Received JSON Data:")
        print(f"   Keys: {list(data.keys())}")
        print(f"   Full Data: {data}")
        print()
        
        # 2. 데이터 파싱 시 디버깅 로그
        print("🔍 Parsing Data...")
        
        product_name = data.get('product_name', '')
        product_type = data.get('product_type', 'overseas')
        country = data.get('country', '')
        city = data.get('city', '')
        departure_port = data.get('departure_port', '')
        nights = data.get('nights')
        days = data.get('days')
        status = data.get('status', 'draft')  # 'draft' or 'published'
        
        # JSON 필드들
        details_json = data.get('details', {})
        hotels_json = data.get('hotels', [])
        golf_courses_json = data.get('golf_courses', [])
        itinerary_options_json = data.get('itinerary_options', [])
        flight_info_json = data.get('flight_info', {})
        ai_content_json = data.get('ai_content', {})
        price_list = data.get('price_info', [])
        
        print(f"   ✓ Product Name: {product_name}")
        print(f"   ✓ Product Type: {product_type}")
        print(f"   ✓ Location: {country} {city}")
        print(f"   ✓ Departure Port: {departure_port}")
        print(f"   ✓ Nights: {nights}, Days: {days}")
        print(f"   ✓ Status: {status}")
        print(f"   ✓ Details JSON: {len(str(details_json))} chars")
        print(f"   ✓ Hotels: {len(hotels_json)} items")
        print(f"   ✓ Golf Courses: {len(golf_courses_json)} items")
        print(f"   ✓ Itinerary Options: {len(itinerary_options_json)} items")
        print(f"   ✓ Price List: {len(price_list)} items")
        
        # 필수 필드 체크
        if not product_name:
            print("❌ ERROR: product_name is required")
            return jsonify({"error": "product_name is required"}), 400
        
        if not details_json:
            print("⚠️  WARNING: details_json is empty")
        
        if not price_list:
            print("⚠️  WARNING: price_list is empty")
        
        print()
        
        # 3. 객체 생성 시 디버깅 로그
        print("📝 Creating Product object...")
        
        product = Product(
            product_name=product_name,
            product_type=product_type,
            country=country,
            city=city,
            departure_port=departure_port,
            nights=nights,
            days=days,
            details_json=details_json,
            hotels_json=hotels_json,
            golf_courses_json=golf_courses_json,
            itinerary_options_json=itinerary_options_json,
            flight_info_json=flight_info_json,
            ai_content_json=ai_content_json,
            status=status
        )
        
        print(f"   ✓ Product object created: {product.product_name}")
        print()
        
        # 4. 가격 저장 시 디버깅 로그
        print(f"💰 Processing {len(price_list)} prices...")
        
        for idx, price_data in enumerate(price_list, 1):
            print(f"   [{idx}/{len(price_list)}] Processing price...")
            
            departure_date_str = price_data.get('departure_date')
            if not departure_date_str:
                print(f"      ⚠️  WARNING: Skipping price {idx} - no departure_date")
                continue
            
            try:
                # 날짜 파싱 (YYYY-MM-DD 형식)
                departure_date = datetime.strptime(departure_date_str, '%Y-%m-%d')
            except ValueError as e:
                print(f"      ❌ ERROR: Invalid date format '{departure_date_str}': {e}")
                continue
            
            price_obj = ProductPrice(
                product=product,
                departure_date=departure_date,
                night_count=price_data.get('night_count'),
                day_count=price_data.get('day_count'),
                group_size=price_data.get('group_size', 1),
                price_adult=price_data.get('price_adult'),
                price_child=price_data.get('price_child'),
                status=price_data.get('status', 'available')
            )
            
            print(f"      ✓ Price object created: {departure_date_str} - {price_data.get('price_adult')}원")
        
        print()
        
        # 5. 커밋 직전 디버깅 로그
        print("💾 Committing to DB...")
        print(f"   Product: {product.product_name}")
        print(f"   Prices: {len(product.prices)} items")
        
        db.add(product)
        db.flush()  # ID를 얻기 위해 flush
        
        print(f"   ✓ Product ID generated: {product.id}")
        
        db.commit()
        
        # 6. 커밋 직후 디버깅 로그
        print("✅ Saved Successfully!")
        print(f"   Product ID: {product.id}")
        print(f"   Product Name: {product.product_name}")
        print(f"   Status: {product.status}")
        print(f"   Created At: {product.created_at}")
        print("="*80 + "\n")
        
        return jsonify({
            "status": "success",
            "message": f"Product saved successfully",
            "product_id": product.id
        }), 201
        
    except Exception as e:
        # 7. 에러 발생 시 상세 디버깅 로그
        print("\n" + "="*80)
        print("❌ ERROR OCCURRED DURING SAVE")
        print("="*80)
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print("\n📋 Full Traceback:")
        print("-"*80)
        traceback.print_exc()
        print("-"*80)
        print("="*80 + "\n")
        
        if db:
            try:
                db.rollback()
                print("   ✓ Transaction rolled back")
            except Exception as rollback_error:
                print(f"   ❌ Rollback failed: {rollback_error}")
        
        return jsonify({
            "status": "error",
            "message": f"Failed to save product: {str(e)}",
            "error_type": type(e).__name__
        }), 500
        
    finally:
        if db:
            try:
                db.close()
                print("   ✓ DB Session closed")
            except Exception as close_error:
                print(f"   ⚠️  Error closing DB session: {close_error}")