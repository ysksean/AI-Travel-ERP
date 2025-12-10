"""
챗봇 검색 로직 테스트 스크립트
실제 DB에서 검색이 제대로 작동하는지 확인
"""
import sys
import os

# app.py의 함수들을 import하기 위해 경로 추가
sys.path.insert(0, os.path.dirname(__file__))

from app import search_products_from_db, parse_query_intent

def test_parse_intent():
    """의도 파싱 테스트"""
    print("=" * 60)
    print("🧪 Testing Query Intent Parsing")
    print("=" * 60)
    
    test_queries = [
        "3박 4일 미야코지마 상품 추천해줘",
        "100만원 이하 일본 여행",
        "홋카이도 4일 패키지",
        "다낭 3박 4일 50만원 이하",
        "제주도 여행"
    ]
    
    for query in test_queries:
        intent = parse_query_intent(query)
        print(f"\n📝 Query: {query}")
        print(f"   Intent: {intent}")

def test_db_search():
    """DB 검색 테스트"""
    print("\n" + "=" * 60)
    print("🔍 Testing DB Product Search")
    print("=" * 60)
    
    test_queries = [
        "3박 4일 미야코지마",
        "홋카이도 여행",
        "100만원 이하 일본",
        "다낭 3박 4일"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 60)
        results = search_products_from_db(query)
        if results:
            print(f"✅ Found products:\n{results}")
        else:
            print("❌ No products found")
        print()

if __name__ == "__main__":
    print("🚀 Starting Chatbot Search Test\n")
    
    # 1. 의도 파싱 테스트
    test_parse_intent()
    
    # 2. DB 검색 테스트
    test_db_search()
    
    print("\n" + "=" * 60)
    print("✅ Test Complete")
    print("=" * 60)

