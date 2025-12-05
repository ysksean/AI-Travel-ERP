import os
import torch
from transformers import ElectraConfig, ElectraForTokenClassification, AutoTokenizer

# ======================================================
# [설정] 모델이 저장될 물리적 위치 (중요!)
# ai_service.py가 바라보는 경로와 일치시켜야 합니다.
# ======================================================
# 현재: flask_web/init_dummy_models.py
# 목표: flask_web 상위 폴더(travel)의 models 폴더
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # flask_web 폴더
MODEL_SAVE_DIR = os.path.join(os.path.dirname(BASE_DIR), 'models')

# 폴더가 없으면 생성
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)
    print(f"📂 모델 메인 폴더 생성: {MODEL_SAVE_DIR}")


def create_dummy_models():
    print(f"🚀 더미 모델 생성을 시작합니다... (저장소: {MODEL_SAVE_DIR})")

    # --------------------------------------------------
    # 1. Tokenizer (토크나이저)
    # --------------------------------------------------
    # 토크나이저는 구조가 복잡하므로, 실제 KoELECTRA 토크나이저를
    # 한 번만 다운로드해서 저장하는 것이 안전합니다.
    print("\n[1/2] Tokenizer 다운로드 및 저장 중...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        save_path = os.path.join(MODEL_SAVE_DIR, 'tokenizer')
        tokenizer.save_pretrained(save_path)
        print(f"  ✅ 저장 완료: {save_path}")
    except Exception as e:
        print(f"  ❌ 실패 (인터넷 연결 확인 필요): {e}")

    # --------------------------------------------------
    # 2. [M1] NER 모델 (KoELECTRA 구조, 랜덤 가중치)
    # --------------------------------------------------
    print("\n[2/2] [M1] NER 모델(Dummy) 생성 중...")

    # 껍데기(설정)만 정의합니다. (가볍게 만들기 위해 레이어 수를 줄임)
    config = ElectraConfig(
        vocab_size=35000,  # KoELECTRA 어휘 크기
        hidden_size=64,  # (더미용) 크기 대폭 축소
        num_hidden_layers=2,  # (더미용) 레이어 2개만
        num_attention_heads=4,
        intermediate_size=256,
        num_labels=7  # B-HOTEL, I-PRICE 등 태그 개수 (0~6)
    )

    # 설정대로 모델 초기화 (랜덤 값)
    model = ElectraForTokenClassification(config)

    # 저장
    save_path = os.path.join(MODEL_SAVE_DIR, 'koelectra_ner')
    model.save_pretrained(save_path)
    print(f"  ✅ 저장 완료: {save_path}")

    print("\n🎉 모든 더미 모델 생성이 완료되었습니다!")
    print(f"이제 ai_service.py를 실행하면 '{MODEL_SAVE_DIR}' 경로에서 모델을 로드합니다.")


if __name__ == "__main__":
    create_dummy_models()