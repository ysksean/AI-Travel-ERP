import json
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, ElectraForTokenClassification
from torch.optim import AdamW

# --- 1. 설정 및 태그 정의 (ERP 폼 구조와 1:1 매핑) ---
EPOCHS = 5
LEARNING_RATE = 5e-5
BATCH_SIZE = 2
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '../models')
DATA_FILE = os.path.join(BASE_DIR, 'train_data.json')

# [중요] 세분화된 태그 리스트 (총 31개)
LABEL_LIST = [
    "O",
    "B-HOTEL_NAME", "I-HOTEL_NAME", "B-HOTEL_GRADE", "I-HOTEL_GRADE", "B-HOTEL_LOC", "I-HOTEL_LOC",
    "B-GOLF_NAME", "I-GOLF_NAME", "B-GOLF_OP", "I-GOLF_OP",
    "B-FLIGHT_NAME", "I-FLIGHT_NAME", "B-FLIGHT_NUM", "I-FLIGHT_NUM", "B-DEPART_TIME", "I-DEPART_TIME",
    "B-PRICE", "I-PRICE", "B-INCLUSION", "I-INCLUSION", "B-EXCLUSION", "I-EXCLUSION",
    "B-REFUND", "I-REFUND", "B-DATE", "I-DATE", "B-CITY", "I-CITY", "B-NOTE", "I-NOTE"
]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}


class NERDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        text = item['text']
        labels = item['labels']

        words = text.split()
        token_list = []
        label_ids = []

        for word, label in zip(words, labels):
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens: continue
            token_list.extend(word_tokens)
            try:
                label_ids.append(LABEL2ID[label])
            except KeyError:
                label_ids.append(0)  # 모르는 라벨은 'O' 처리
            label_ids.extend([-100] * (len(word_tokens) - 1))

        encoding = self.tokenizer.encode_plus(
            token_list, max_length=self.max_len, padding='max_length',
            truncation=True, is_split_into_words=True, return_tensors='pt'
        )

        pad_len = self.max_len - len(label_ids)
        if pad_len > 0:
            label_ids += [-100] * pad_len
        else:
            label_ids = label_ids[:self.max_len]

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }


def train():
    print("🚀 고도화 모델 학습 준비 중...")

    # 저장 폴더가 없으면 생성
    ner_save_path = os.path.join(MODEL_DIR, 'koelectra_ner')
    if not os.path.exists(ner_save_path): os.makedirs(ner_save_path)

    tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

    # [핵심] num_labels를 31개로 설정하여 모델 초기화
    model = ElectraForTokenClassification.from_pretrained(
        "monologg/koelectra-base-v3-discriminator",
        num_labels=len(LABEL_LIST)
    )

    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    dataset = NERDataset(raw_data, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"🔥 학습 시작! (Device: {device}, Labels: {len(LABEL_LIST)})")

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch + 1}/{EPOCHS} - Loss: {total_loss / len(loader):.4f}")

    model.save_pretrained(ner_save_path)
    tokenizer.save_pretrained(os.path.join(MODEL_DIR, 'tokenizer'))
    print(f"\n🎉 학습 완료! 고도화된 모델이 '{ner_save_path}'에 저장되었습니다.")


if __name__ == "__main__":
    train()