class RagService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RagService, cls).__new__(cls)
            # 진짜 모델 로딩은 생략 (B는 이게 필요 없음)
            print("🤖 [System] Mock RAG Service Started (No GPU needed)")
        return cls._instance

    def initialize(self):
        pass

    def add_knowledge(self, text_list: list[str]):
        # 데이터 저장하는 척만 함
        print(f"📥 [Mock] {len(text_list)} documents received but not saved.")

    def search(self, query: str, top_k: int = 3) -> str:
        # ★ 여기가 중요합니다 ★
        # 실제 검색 대신 무조건 아래 텍스트를 리턴합니다.
        # 나중에 Member A가 이 함수 내용만 진짜로 바꾸면 됩니다.

        return """
        [검색된 여행 상품 정보]
        1. 상품명: 미야코지마 힐링 골프 3박 4일
           - 가격: 129만원 (항공권 포함)
           - 호텔: 힐튼 오키나와 (5성급, 오션뷰)
           - 골프장: 에메랄드 코스트 CC, 시기라 베이 CC
           - 특전: 전 일정 조식/석식 포함, 송영 차량 제공
           - 출발일: 매일 출발 가능

        2. 상품명: 후쿠오카 가성비 골프 2박 3일
           - 가격: 89만원
           - 특징: 시내 접근성 좋음, 노캐디 플레이
        """


# 전역에서 쓸 수 있게 인스턴스 미리 생성
rag_engine = RagService()