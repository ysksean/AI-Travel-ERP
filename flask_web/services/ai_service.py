import os
import torch
from transformers import AutoTokenizer, ElectraForTokenClassification
from services.parsing_service import parsing_manager

# 태그 라벨 정의
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

    # 싱글톤 패턴 적용 (인스턴스 중복 생성 방지)
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AIService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        # 이미 초기화되었다면 건너뜀 (서버 재시작/리로드 시 안전장치)
        if getattr(self, '_initialized', False):
            return

        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.environ.get('MODEL_DIR', os.path.join(self.base_dir, '../models'))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.models = {}
        self.tokenizer = None

        # [핵심] __init__에서는 모델을 로드하지 않음 (Lazy Loading 준비)
        # 이렇게 해야 'import ai_service' 할 때 시간이 걸리지 않고 즉시 임포트됨
        self._resources_loaded = False
        self._initialized = True

    def _ensure_resources_loaded(self):
        """실제 기능(함수)이 호출될 때 비로소 모델을 로드함 (Lazy Loading)"""
        if self._resources_loaded:
            return

        print(f"🚀 AI 리소스 지연 로딩 시작 (Device: {self.device})")

        # 1. 토크나이저 로드
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

        # 2. NER 모델 로드
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

        # 로딩 완료 플래그 설정
        self._resources_loaded = True

    def extract_quotation_info(self, file_path):
        """
        파일을 파싱하고 AI로 정보를 추출함.
        """
        # [지연 로딩 트리거] 이 함수가 실행될 때 모델이 없으면 로드함
        self._ensure_resources_loaded()

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

        # 모델 로드 실패했거나 파일이 없으면 추론 건너뜀
        if 'ner' not in self.models or self.tokenizer is None:
            print("⚠️ AI 모델이 준비되지 않아 자동 추출을 건너뜁니다.")
            ai_status = "skipped_no_model"
            extracted_tags = {}
        else:
            try:
                extracted_tags = self._run_ner_inference(raw_text)
            except Exception as e:
                print(f"❌ 추론 도중 에러 발생: {e}")
                ai_status = "inference_error"
                extracted_tags = {}

        # 3. 폼 매핑
        form_data = self._map_to_form(extracted_tags)

        # AI가 동작하지 않았을 경우 처리
        if ai_status != "success":
            form_data["ai_content"]["body_text"] = raw_text[:3000]
            form_data["details"]["references"] = "⚠️ AI 모델이 없어 자동 분석되지 않았습니다. 원본 텍스트를 참고하세요."

        return {
            "status": "success",
            "ai_status": ai_status,
            "file_name": os.path.basename(file_path),
            "data": form_data,
            "raw_data": extracted_tags
        }

    def _run_ner_inference(self, text):
        if not text: return {}

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
        form = self._get_default_form()

        if not tags:
            return form

        if tags.get("CITY"): form["location_info"]["city"] = tags["CITY"][0]
        if tags.get("HOTEL_NAME"):
            form["hotels"][0]["name_kr"] = tags["HOTEL_NAME"][0]
            form["product_info"]["product_name"] = f"{tags['HOTEL_NAME'][0]} 프리미엄 패키지"
        if tags.get("HOTEL_GRADE"): form["hotels"][0]["grade"] = tags["HOTEL_GRADE"][0]
        if tags.get("HOTEL_LOC"): form["hotels"][0]["location"] = tags["HOTEL_LOC"][0]

        if tags.get("GOLF_NAME"): form["golf_courses"][0]["name_kr"] = tags["GOLF_NAME"][0]
        if tags.get("GOLF_OP"): form["golf_courses"][0]["operation_info"] = ", ".join(tags["GOLF_OP"])

        if tags.get("FLIGHT_NAME"): form["flight_info"]["airline"] = tags["FLIGHT_NAME"][0]
        if tags.get("FLIGHT_NUM"): form["flight_info"]["flight_number"] = tags["FLIGHT_NUM"][0]
        if tags.get("DEPART_TIME"): form["flight_info"]["departure_time"] = tags["DEPART_TIME"][0]

        if tags.get("DATE"): form["product_info"]["event_period"]["start_date"] = tags["DATE"][0]
        if tags.get("INCLUSION"): form["details"]["inclusions"] = tags["INCLUSION"]
        if tags.get("EXCLUSION"): form["details"]["exclusions"] = tags["EXCLUSION"]
        if tags.get("REFUND"): form["policies"]["cancellation_refund"] = " ".join(tags["REFUND"])
        if tags.get("NOTE"): form["details"]["references"] = "\n".join(tags["NOTE"])

        if tags.get("PRICE"):
            price_txt = ", ".join(tags["PRICE"])
            form["details"]["others"] = f"추출 가격: {price_txt}"

        return form


# [수정 완료] 아래 변수명을 'ai_service'로 설정하여
# from services.ai_service import ai_service 구문이 작동하게 함
ai_service = AIService()