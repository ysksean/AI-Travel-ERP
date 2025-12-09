# services/parsing_service.py

import os
import time
import json
import pandas as pd
import pdfplumber
from docx import Document
import google.generativeai as genai
from pdf2image import convert_from_path

# [수정됨] typing_extensions에서 TypedDict를 가져와야 Python 3.11 이하에서도 에러가 안 남
from typing_extensions import TypedDict
import typing  # List, Optional 등을 위해 유지

# [환경 설정]
# .env 파일에서 키를 로드합니다.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.5-flash-lite"  # 혹은 1.5-flash

# Poppler 경로 설정
DEFAULT_POPPLER_PATH = r"C:\poppler\Library\bin"
POPPLER_BIN_PATH = os.getenv("POPPLER_BIN_PATH", DEFAULT_POPPLER_PATH)

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

try:
    from docx2pdf import convert as docx_to_pdf_tool
except ImportError:
    docx_to_pdf_tool = None


# ==========================================
# [스키마 정의] (수정됨: TypedDict 사용)
# ==========================================

# [수정] typing.TypedDict -> TypedDict 로 변경
class BasicInfo(TypedDict):
    product_type: str | None
    is_flight_included: bool | None
    is_vat_included: bool | None


class LocationInfo(TypedDict):
    country: str | None
    city: str | None
    departure_port: str | None


class EventPeriod(TypedDict):
    available_days: list[str]


class ProductInfo(TypedDict):
    product_name: str | None
    event_period: EventPeriod


class MetaInfoHotel(TypedDict):
    check_in_out: str | None
    website: str | None


class Hotel(TypedDict):
    name_kr: str | None
    description: str | None
    meta_info: MetaInfoHotel


class MetaInfoGolf(TypedDict):
    detail_info: str | None
    website: str | None
    hole_info: str | None  # 예: "18홀/72파/6912Y"


class GolfCourse(TypedDict):
    name_kr: str | None
    address: str | None  # 주소 (AI 지식 기반으로 채움)
    operation_info: str | None
    description: str | None  # 골프장 설명
    meta_info: MetaInfoGolf


class TouristSpot(TypedDict):
    name: str | None


class Details(TypedDict):
    inclusions: list[str]
    exclusions: list[str]
    others: str | None
    is_insurance_included: bool | None
    is_guide_included: bool | None
    special_notes: list[str]


class AiContent(TypedDict):
    body_text: str | None


class FlightInfo(TypedDict):
    airline: str | None
    flight_number: str | None
    departure_time: str | None
    arrival_time: str | None


class PriceInfo(TypedDict):
    departure_date: str
    night_count: int
    day_count: int
    group_size: int
    price_adult: int
    status: str


class DailyMeal(TypedDict):
    breakfast: str | None  # "포함", "불포함", "호텔식", "기내식" 등
    lunch: str | None
    dinner: str | None


class DailySchedule(TypedDict):
    day: int  # 1, 2, 3...
    transport: str | None  # "전용택시", "항공", "버스" 등
    time: str | None  # "08:00", "14:30" 등
    description: str | None  # 일정 상세 설명
    meals: DailyMeal | None


class ItineraryOption(TypedDict):
    option_name: str  # 예: "2박 3일", "3박 4일", "기본 일정"
    schedules: list[DailySchedule]  # 해당 옵션의 일정 리스트


# ★ 메인 스키마
class TravelProductSchema(TypedDict):
    basic_info: BasicInfo
    location_info: LocationInfo
    product_info: ProductInfo
    hotels: list[Hotel]
    golf_courses: list[GolfCourse]
    tourist_spots: list[TouristSpot]
    details: Details
    ai_content: AiContent
    flight_info: FlightInfo
    price_info: list[PriceInfo]
    itinerary_options: list[ItineraryOption]  # [변경] 기간별 일정 옵션 리스트


# 가격표 전용 스키마
class PriceListSchema(TypedDict):
    prices: list[PriceInfo]


# ==========================================
# [클래스] 만능 여행 데이터 처리기
# ==========================================
class UniversalTravelAI:
    def __init__(self):
        if not GOOGLE_API_KEY:
            print("⚠️ Warning: GOOGLE_API_KEY not found.")
        self.model = genai.GenerativeModel(MODEL_NAME)

    def _generate_with_retry(self, content, config, retries=3):
        for i in range(retries):
            try:
                return self.model.generate_content(
                    content, generation_config=config, request_options={"timeout": 120}
                )
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    wait_time = (i + 1) * 5
                    print(f"   ⚠️ Quota Exceeded. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise e
        return None

    def _extract_text_content(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        print(f"   📄 [Text Extractor] Reading {ext} file...")
        try:
            if ext == '.pdf':
                text = ""
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text: text += page_text + "\n"
                return text
            elif ext in ['.docx', '.doc']:
                doc = Document(file_path)
                return "\n".join([p.text for p in doc.paragraphs])
            elif ext in ['.xlsx', '.xls']:
                xls = pd.read_excel(file_path, sheet_name=None)
                text = ""
                for sheet, df in xls.items():
                    text += f"--- Sheet: {sheet} ---\n"
                    try:
                        text += df.to_markdown(index=False) + "\n"
                    except:
                        text += df.to_string(index=False) + "\n"
                return text
            elif ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            return ""
        except Exception as e:
            print(f"   ❌ Text Extraction Failed: {e}")
            return ""

    def _convert_to_images(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        print(f"   🖼️ [Image Converter] Processing {ext} file...")

        pdf_path = file_path
        if ext in ['.docx', '.doc']:
            try:
                from docx2pdf import convert
                pdf_path = os.path.splitext(file_path)[0] + ".pdf"
                convert(file_path, pdf_path)
            except:
                return []

        if pdf_path.lower().endswith('.pdf'):
            try:
                use_poppler_path = POPPLER_BIN_PATH if os.name == 'nt' else None
                return convert_from_path(pdf_path, dpi=300, poppler_path=use_poppler_path)
            except Exception as e:
                print(f"   ❌ PDF to Image failed: {e}")
                return []
        return []

    # ---------------------------------------------------------
    # 메인 분석 메서드
    # ---------------------------------------------------------
    def analyze(self, product_file, price_file=None):
        print(f"\n🚀 Analysis Started: {product_file}")

        # 1. 텍스트 추출
        product_text = self._extract_text_content(product_file)
        if not product_text:
            return {"error": "상품 텍스트 추출 실패"}

        # 2. Gemini 상품 분석
        product_data = self._call_gemini_product(product_text)

        # 3. 가격 분석
        price_source = price_file if price_file else product_file

        if not price_source or not os.path.exists(price_source):
            if not product_data: product_data = {}
            product_data['price_info'] = []
            return product_data

        price_ext = os.path.splitext(price_source)[1].lower()
        price_list = []

        if price_ext in ['.xlsx', '.xls', '.csv', '.txt']:
            print("   -> [Strategy A] Excel/Text detected.")
            raw_text = self._extract_text_content(price_source)
            price_list = self._call_gemini_price_text(raw_text)
        else:
            print("   -> [Strategy B] PDF/Image detected.")
            images = self._convert_to_images(price_source)
            if images:
                price_list = self._call_gemini_price_vision(images)
            else:
                print("   ⚠️ Vision failed. Fallback to Text Analysis.")
                raw_text = self._extract_text_content(price_source)
                price_list = self._call_gemini_price_text(raw_text)

        if not product_data: product_data = {}
        product_data['price_info'] = price_list

        return product_data

    # ---------------------------------------------------------
    # Gemini Prompt
    # ---------------------------------------------------------
    def _call_gemini_product(self, text):
        prompt = """
        You are a generic travel product data parser.
        Analyze the text and extract details into the JSON schema perfectly.

        [MAPPING RULES - CRITICAL]

        1. **Multiple Options Handling (Hotels/Golf)**:
           - If the text lists multiple options (e.g., "3-star: A Hotel, 4-star: B Hotel" or "A CC, B CC, C CC"),
             **extract ALL of them** as separate items in the `hotels` or `golf_courses` list.
           - Do not merge them into one string. Create a list of objects.

        2. **Itinerary Parsing - CRITICAL EXTRACTION RULES**:
           - **Check if the PDF contains MULTIPLE versions of schedule tables** (e.g., "2박 3일" schedule, "3박 4일" schedule).
           - **Separate and extract each schedule table** into different `ItineraryOption` objects.
           - For each schedule option:
             * `option_name`: Set to the period name found in the document (e.g., "2박 3일", "3박 4일", "4박 5일").
             * If only ONE schedule table exists, set `option_name` to "기본 일정" and extract it as a single option.
             * `schedules`: Array of `DailySchedule` objects for that option.
           - For each day in a schedule, extract:
             * `day`: Day number (1, 2, 3...)
             * `transport`: Transportation method (e.g., "전용택시", "항공", "버스", "셔틀")
             * `time`: Time information (e.g., "08:00", "14:30")
             * `description`: Detailed description of the day's activities
             * `meals`: Object with `breakfast`, `lunch`, `dinner` fields (values: "포함", "불포함", "호텔식", "기내식", "클럽식", "자유식" 등)
           - If the document has a table format, parse each row as a day entry.
           - **IMPORTANT**: If you find multiple schedule tables with different periods (e.g., "2박 3일 일정표", "3박 4일 일정표"), create separate `ItineraryOption` entries for each.

        3. **Multiple Golf Courses**:
           - Extract ALL golf courses mentioned in the document.
           - Each golf course should be a separate object in the `golf_courses` list.
           - Look for patterns like "에메랄드 CC", "시기라 베이", "18홀 라운딩" etc.

        4. **Golf Details**:
           - For each golf course, extract:
             * `name_kr`: Golf course name (Korean or local name)
             * `address`: Use your world knowledge to fill in the address based on the golf course name and location context (e.g., "일본 오키나와현 미야코지마시...")
             * `operation_info`: Operating information (e.g., "티오프 7분 간격, 카트 필수")
             * `description`: Description of the golf course (e.g., "바다와 맞닿은 아름다운 코스...")
             * `meta_info.hole_info`: Hole information (e.g., "18홀/72파/6912Y")

        5. **Flight Info**:
           - Look for flight codes (e.g., LJ357, KE463, 7C, OZ) and times.
           - If found, fill `flight_info` and set `is_flight_included` = True.

        6. **Location**:
           - If text contains "오키나와" or "미야코지마", set Country="일본".
           - If text contains "베트남" or "다낭", set Country="베트남".

        7. **AI Content Creation**:
           - Read the whole text and write a summarizing marketing text in `ai_content.body_text`.
           - It should highlight key selling points (e.g., hotel grade, golf course view).

        8. **Language**:
           - All text output MUST be in **Korean**.
        """
        try:
            resp = self._generate_with_retry(
                [prompt, text[:50000]],
                genai.GenerationConfig(response_mime_type="application/json", response_schema=TravelProductSchema)
            )
            if resp:
                data = json.loads(resp.text)
                country = data.get("location_info", {}).get("country", "")
                if country and any(x in country for x in ["한국", "대한민국", "제주", "Korea"]):
                    data["basic_info"]["product_type"] = "국내상품"
                else:
                    data["basic_info"]["product_type"] = "해외상품"
                return data
            return {}
        except Exception as e:
            print(f"   ❌ Product Info Error: {e}")
            return {}

    def _call_gemini_price_text(self, text):
        prompt = "Extract pricing table. Rows: Date, Price, Headcount. Date format: YYYY-MM-DD."
        try:
            resp = self._generate_with_retry(
                [prompt, text[:30000]],
                genai.GenerationConfig(response_mime_type="application/json", response_schema=PriceListSchema)
            )
            return json.loads(resp.text).get('prices', []) if resp else []
        except:
            return []

    def _call_gemini_price_vision(self, images):
        all_prices = []
        for img in images[:5]:
            try:
                time.sleep(1.5)
                resp = self._generate_with_retry(
                    ["Extract price table. Output JSON only.", img],
                    genai.GenerationConfig(response_mime_type="application/json", response_schema=PriceListSchema)
                )
                if resp: all_prices.extend(json.loads(resp.text).get('prices', []))
            except:
                continue
        return all_prices