import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import os


class ModelLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        print("\n⚡ [System] Initializing AI Models... (첫 실행 시 다운로드로 인해 시간이 걸립니다)")

        # GPU 사용 가능 여부 확인 (NVIDIA 그래픽카드 있으면 cuda, 없으면 cpu)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   ⚙️  Device set to: {self.device.upper()}")

        # 1. 임베딩 모델 (검색용) - Ko-SBERT
        print("   📥 Loading Embedding Model (Ko-SBERT)...")
        self.embedding_model = SentenceTransformer('jhgan/ko-sroberta-multitask')

        # 2. 의도 분류/판별 모델 - KoELECTRA
        # (여기서는 기본 모델을 로드합니다. 추후 Fine-tuning된 가중치로 교체 가능)
        print("   📥 Loading Intent Model (KoELECTRA)...")
        self.electra_tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.electra_model = AutoModelForSequenceClassification.from_pretrained(
            "monologg/koelectra-base-v3-discriminator").to(self.device)

        # 3. 요약 모델 - KoBART
        print("   📥 Loading Summary Model (KoBART)...")
        self.bart_tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-summarization")
        self.bart_model = AutoModelForSeq2SeqLM.from_pretrained("gogamza/kobart-summarization").to(self.device)

        print("✅ [System] All Local Models Loaded Successfully!\n")

    # --- Getter 메서드들 ---
    def get_embedding_model(self):
        return self.embedding_model

    def get_electra(self):
        return self.electra_tokenizer, self.electra_model

    def get_bart(self):
        return self.bart_tokenizer, self.bart_model


# 테스트 실행 코드 (이 파일을 직접 실행했을 때만 동작)
if __name__ == "__main__":
    loader = ModelLoader()
    print("모델 로더 테스트 완료.")