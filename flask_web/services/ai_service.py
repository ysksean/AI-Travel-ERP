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
        self.model_dir = os.environ.get('MODEL_DIR', os.path.join(self.base_dir, '../models'))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.tokenizer = None
        self.load_resources()
        self._initialized = True

    def load_resources(self):
        print(f"🚀 AI 서비스 로딩 (Device: {self.device})")
        try:
            tok_path = os.path.join(self.model_dir, 'tokenizer')
            if os.path.exists(tok_path):
                self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

            m1_path = os.path.join(self.model_dir, 'koelectra_ner')
            if os.path.exists(m1_path):
                self.models['ner'] = ElectraForTokenClassification.from_pretrained(m1_path).to(self.device)
                self.models['ner'].eval()
                print("  ✅ [M1] 고도화된 NER 모델 로드 완료")
            else:
                print(f"  ⚠️ 모델 없음: {m1_path}")
        except Exception as e:
            print(f"❌ 로딩 에러: {e}")

    def extract_quotation_info(self, file_path):
        raw_text = parsing_manager.parse_file(file_path)
        if not raw_text: return {"status": "error", "message": "텍스트 추출 실패"}
        if 'ner' not in self.models: return {"status": "warning", "raw_text": raw_text[:200]}

        # AI 추론
        extracted_tags = self._run_ner_inference(raw_text)
        # 폼 매핑
        form_data = self._map_to_form(extracted_tags)

        return {
            "status": "success",
            "file_name": os.path.basename(file_path),
            "data": form_data,
            "raw_data": extracted_tags
        }

    def _run_ner_inference(self, text):
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

    def _map_to_form(self, tags):
        """ [매핑 엔진] 추출된 태그를 ERP 폼 구조에 정확히 배치 """
        form = {
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

        # 1. 지역 및 호텔
        if tags.get("CITY"): form["location_info"]["city"] = tags["CITY"][0]
        if tags.get("HOTEL_NAME"):
            form["hotels"][0]["name_kr"] = tags["HOTEL_NAME"][0]
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

        # 5. 가격 (별도 필드 없으면 기타란에)
        if tags.get("PRICE"):
            price_txt = ", ".join(tags["PRICE"])
            form["details"]["others"] = f"추출 가격: {price_txt}"

        return form


ai_manager = AIService()