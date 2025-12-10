import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from models.loader import ModelLoader


class RagService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RagService, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        # ModelLoader를 통해 임베딩 모델 가져오기
        self.loader = ModelLoader()

        # 데이터 저장소 (In-Memory)
        self.documents = []  # 실제 텍스트 저장 리스트
        self.vectors = None  # 벡터값 저장 (Numpy Array)
        print("📚 [RAG Service] Search Engine Ready.")

    def add_knowledge(self, text_list: list[str]):
        """
        새로운 정보를 학습(메모리에 저장)시킵니다.
        """
        if not text_list:
            return

        print(f"   🔄 Indexing {len(text_list)} documents...")
        model = self.loader.get_embedding_model()

        # 1. 텍스트 -> 벡터 변환 (Encoding)
        new_vectors = model.encode(text_list)

        # 2. 기존 데이터와 합치기
        self.documents.extend(text_list)

        if self.vectors is None:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])

        print(f"   ✅ Total Indexed Documents: {len(self.documents)}")

    def search(self, query: str, top_k: int = 3) -> str:
        """
        사용자 질문(query)과 가장 유사한 문서를 찾아 반환합니다.
        """
        if self.vectors is None or len(self.vectors) == 0:
            return "아직 학습된 여행 정보가 없습니다."

        model = self.loader.get_embedding_model()

        # 1. 질문을 벡터로 변환
        query_vector = model.encode([query])  # Shape: (1, 768)

        # 2. 코사인 유사도 계산 (질문 vs 저장된 모든 문서)
        scores = cosine_similarity(query_vector, self.vectors)[0]

        # 3. 점수가 높은 순서대로 정렬하여 인덱스 뽑기
        top_indices = np.argsort(scores)[::-1][:top_k]

        # 4. 결과 텍스트 조합
        results = []
        for idx in top_indices:
            score = scores[idx]
            # 유사도 점수가 너무 낮으면(0.3 미만) 엉뚱한 정보이므로 제외
            if score > 0.3:
                results.append(f"[관련도 {score:.2f}] {self.documents[idx]}")

        if not results:
            return "관련된 정보를 찾을 수 없습니다."

        return "\n\n".join(results)


# 전역 인스턴스 생성 (다른 파일에서 import rag_engine 으로 사용)
rag_engine = RagService()

# ==========================================
# ★ [Member A 자체 테스트용 코드] ★
# 이 파일을 직접 실행하면 RAG 동작을 검증할 수 있습니다.
# ==========================================
if __name__ == "__main__":
    print("🚀 RAG 엔진 자체 테스트를 시작합니다...")

    # 1. 엔진 초기화 (모델 로드 포함)
    engine = RagService()

    # 2. 가짜 데이터 주입 (DB 대신 리스트로 테스트)
    test_data = [
        "미야코지마 골프 패키지: 2025년 11월 출발, 3박 4일, 가격 120만원, 오션뷰 호텔 포함.",
        "제주도 2인 골프: 노캐디 가능, 카트비 별도, 주중 15만원.",
        "오키나와 본섬: 북부 골프장 이용 시 렌터카 필수, 4인 기준 1인 80만원."
    ]
    engine.add_knowledge(test_data)

    # 3. 검색 테스트
    query = "미야코지마 가격 얼마야?"
    print(f"\n❓ 질문: {query}")
    answer = engine.search(query)
    print(f"💡 검색 결과:\n{answer}")

    print("\n--------------------------------")

    query2 = "제주도 노캐디 돼?"
    print(f"❓ 질문: {query2}")
    answer2 = engine.search(query2)
    print(f"💡 검색 결과:\n{answer2}")