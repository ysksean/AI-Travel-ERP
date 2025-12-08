import os
import torch
from transformers import AutoTokenizer, ElectraForTokenClassification
from services.parsing_service import parsing_manager

LABEL_LIST = [
    "O",
    "B-HOTEL_NAME", "I-HOTEL_NAME", "B-HOTEL_GRADE", "I-HOTEL_GRADE", "B-HOTEL_LOC", "I-HOTEL_LOC",
    "B-GOLF_NAME", "I-GOLF_NAME", "B-GOLF_OP", "I-GOLF_OP",
    "B-FLIGHT_NAME", "I-FLIGHT_NAME", "B-FLIGHT_NUM", "I-FLIGHT_NUM", "B-DEPART_TIME", "I-DEPART_TIME",
    "B-PRICE", "I-PRICE", "B-INCLUSION", "I-INCLUSION", "B-EXCLUSION", "I-EXCLUSION",
    "B-REFUND", "I-REFUND", "B-DATE", "I-DATE", "B-CITY", "I-CITY", "B-NOTE", "I-NOTE"
]
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}


class AIService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AIService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized: return
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        # 환경변수가 없으면 기본 경로 사용
        self.model_dir = os.environ.get('MODEL_DIR', os.path.join(self.base_dir, '../models'))
        # CUDA가 없으면 CPU 사용
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.tokenizer = None
        self.load_resources()
        self._initialized = True

    def load_resources(self):
        print(f"🚀 AI 서비스 로딩 시작 (Device: {self.device})")

        # 1. 토크나이저 로드 (실패해도 서버는 켜져야 함)
        try:
            tok_path = os.path.join(self.model_dir, 'tokenizer')
            if os.path.exists(tok_path):
                self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
                print("  ✅ 로컬 토크나이저 로드 완료")
            else:
                print("  ⚠️ 로컬 토크나이저 없음. HuggingFace 다운로드 시도...")
                self.tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
                print("  ✅ 온라인 토크나이저 로드 완료")
        except Exception as e:
            print(f"  ❌ 토크나이저 로드 실패 (AI 기능 제한됨): {e}")
            self.tokenizer = None

        # 2. NER 모델 로드 (실패해도 서버는 켜져야 함)
        try:
            m1_path = os.path.join(self.model_dir, 'koelectra_ner')
            if os.path.exists(m1_path):
                self.models['ner'] = ElectraForTokenClassification.from_pretrained(m1_path).to(self.device)
                self.models['ner'].eval()
                print("  ✅ [M1] 고도화된 NER 모델 로드 완료")
            else:
                print(f"  ⚠️ 모델 파일 없음: {m1_path} (AI 기능 없이 실행됩니다)")
        except Exception as e:
            print(f"  ❌ 모델 로딩 중 에러 발생: {e}")
            # 모델 변수에 아무것도 넣지 않음으로써 로딩 실패 처리

    def extract_quotation_info(self, file_path):
        """
        파일을 파싱하고 AI로 정보를 추출함.
        모델이 없으면 파싱된 텍스트만 '기타 정보'에 넣어서 반환.
        """
        # 1. 텍스트 추출 (Parsing Service)
        try:
            raw_text = parsing_manager.parse_file(file_path)
            if not raw_text:
                return {"status": "error", "message": "파일에서 텍스트를 읽을 수 없습니다."}
        except Exception as e:
            return {"status": "error", "message": f"파싱 에러: {str(e)}"}

        # 2. 모델 상태 확인 및 추론
        extracted_tags = {}
        ai_status = "success"

        # 모델이나 토크나이저가 없으면 추론 건너뜀
        if 'ner' not in self.models or self.tokenizer is None:
            print("⚠️ AI 모델이 로드되지 않아 자동 추출을 건너뜁니다.")
            ai_status = "skipped_no_model"
            # 모델이 없을 땐 원본 텍스트를 통째로 '기타 정보'나 '본문'에 넣어주는 것이 UX상 좋음
            extracted_tags = {}
        else:
            try:
                extracted_tags = self._run_ner_inference(raw_text)
            except Exception as e:
                print(f"❌ 추론 도중 에러 발생: {e}")
                ai_status = "inference_error"
                extracted_tags = {}

        # 3. 폼 매핑 (태그가 비어있으면 기본 폼 반환)
        form_data = self._map_to_form(extracted_tags)

        # 모델이 없어서 추론을 못했으면, 원본 텍스트를 '본문 내용'에라도 넣어줌 (사용자 편의)
        if ai_status != "success":
            form_data["ai_content"]["body_text"] = raw_text[:3000]  # 너무 길면 자름
            form_data["details"]["references"] = "⚠️ AI 모델이 없어 자동 분석되지 않았습니다. 원본 텍스트를 참고하세요."

        return {
            "status": "success",  # UI에서 에러창 대신 폼을 띄우기 위해 success로 반환
            "ai_status": ai_status,
            "file_name": os.path.basename(file_path),
            "data": form_data,
            "raw_data": extracted_tags
        }

    def _run_ner_inference(self, text):
        if not text: return {}

        # 텍스트가 너무 길면 잘라서 처리하거나 앞부분만 처리 (여기선 512 토큰 제한)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            outputs = self.models['ner'](**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        preds = predictions[0].cpu().numpy()

        results = {}
        current_entity = None
        current_word = ""

        for token, pred_idx in zip(tokens, preds):
            if token in ["[CLS]", "[SEP]", "[PAD]"]: continue

            # ID2LABEL 보호 조치 (인덱스 에러 방지)
            label = ID2LABEL.get(pred_idx, 'O')
            clean_token = token.replace("##", "")

            if label.startswith("B-"):
                if current_entity: results.setdefault(current_entity, []).append(current_word)
                current_entity = label.split("-")[1]
                current_word = clean_token
            elif label.startswith("I-") and current_entity == label.split("-")[1]:
                current_word += clean_token
            else:
                if current_entity:
                    results.setdefault(current_entity, []).append(current_word)
                    current_entity = None
                    current_word = ""

        if current_entity: results.setdefault(current_entity, []).append(current_word)
        return results

    def _get_default_form(self):
        """기본 빈 폼 데이터 구조 반환"""
        return {
            "basic_info": {"product_type": "overseas", "is_flight_included": True, "is_vat_included": True},
            "location_info": {"country": "", "city": "", "departure_port": "ICN"},
            "product_info": {"product_name": "", "itinerary_id": None,
                             "event_period": {"start_date": "", "end_date": "", "available_days": []}},
            "hotels": [{"name_kr": "", "name_en": "", "location": "", "grade": "", "images": [], "description": "",
                        "facilities": [],
                        "meta_info": {"check_in_out": "", "distance_from_city": "", "website": "", "phone": "",
                                      "notice": "", "extra_info": ""}}],
            "golf_courses": [{"name_kr": "", "name_local": "", "images": [], "location": "", "operation_info": "",
                              "meta_info": {"website": "", "phone": "", "detail_info": ""}}],
            "tourist_spots": [],
            "policies": {"safety_rules": "", "cancellation_refund": ""},
            "details": {"inclusions": [], "exclusions": [], "others": "", "is_insurance_included": False,
                        "is_guide_included": True, "special_notes": [], "references": "", "key_points": []},
            "ai_content": {"body_text": "", "detailed_description": ""},
            "flight_info": {"airline": "", "flight_number": "", "departure_time": "", "arrival_time": ""},
            "images": {"thumbnail": "", "body_images": []}
        }

    def _map_to_form(self, tags):
        """ [매핑 엔진] 추출된 태그를 ERP 폼 구조에 정확히 배치 """
        # 기본 폼 로드
        form = self._get_default_form()

        if not tags:
            return form

        # 1. 지역 및 호텔
        if tags.get("CITY"): form["location_info"]["city"] = tags["CITY"][0]
        if tags.get("HOTEL_NAME"):
            form["hotels"][0]["name_kr"] = tags["HOTEL_NAME"][0]
            # 상품명이 비어있을 때만 호텔명으로 자동 채움
            form["product_info"]["product_name"] = f"{tags['HOTEL_NAME'][0]} 프리미엄 패키지"
        if tags.get("HOTEL_GRADE"): form["hotels"][0]["grade"] = tags["HOTEL_GRADE"][0]
        if tags.get("HOTEL_LOC"): form["hotels"][0]["location"] = tags["HOTEL_LOC"][0]

        # 2. 골프장
        if tags.get("GOLF_NAME"): form["golf_courses"][0]["name_kr"] = tags["GOLF_NAME"][0]
        if tags.get("GOLF_OP"): form["golf_courses"][0]["operation_info"] = ", ".join(tags["GOLF_OP"])

        # 3. 항공
        if tags.get("FLIGHT_NAME"): form["flight_info"]["airline"] = tags["FLIGHT_NAME"][0]
        if tags.get("FLIGHT_NUM"): form["flight_info"]["flight_number"] = tags["FLIGHT_NUM"][0]
        if tags.get("DEPART_TIME"): form["flight_info"]["departure_time"] = tags["DEPART_TIME"][0]

        # 4. 기타 정보
        if tags.get("DATE"): form["product_info"]["event_period"]["start_date"] = tags["DATE"][0]
        if tags.get("INCLUSION"): form["details"]["inclusions"] = tags["INCLUSION"]
        if tags.get("EXCLUSION"): form["details"]["exclusions"] = tags["EXCLUSION"]
        if tags.get("REFUND"): form["policies"]["cancellation_refund"] = " ".join(tags["REFUND"])
        if tags.get("NOTE"): form["details"]["references"] = "\n".join(tags["NOTE"])

        # 5. 가격 (별도 필드 없으면 기타란에)
        if tags.get("PRICE"):
            price_txt = ", ".join(tags["PRICE"])
            form["details"]["others"] = f"추출 가격: {price_txt}"

        return form


# [수정됨] 여기서 인스턴스 이름을 routes/reservation.py가 찾는 이름(ai_service)으로 맞춤
ai_service = AIService()