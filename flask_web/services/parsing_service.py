import os
import pandas as pd
import pdfplumber
import docx  # python-docx


class ParsingService:
    def parse_file(self, file_path):
        """
        [진입점] 파일 경로를 받아서 확장자에 맞는 파서로 텍스트 추출
        """
        if not os.path.exists(file_path):
            return ""

        # 확장자 소문자로 추출
        ext = file_path.split('.')[-1].lower()

        try:
            if ext in ['xlsx', 'xls']:
                return self._parse_excel(file_path)
            elif ext == 'pdf':
                return self._parse_pdf(file_path)
            elif ext == 'docx':
                return self._parse_word(file_path)
            # [추가됨] 텍스트 파일 (.txt) 처리
            elif ext == 'txt':
                return self._parse_txt(file_path)
            else:
                return "지원하지 않는 파일 형식입니다."
        except Exception as e:
            print(f"❌ 파일 파싱 실패 ({file_path}): {e}")
            return ""

    # ---------------------------------------------------------
    # 각 파일별 상세 로직
    # ---------------------------------------------------------

    def _parse_txt(self, file_path):
        """ 
        [New] TXT/카카오톡 대화 내용 파싱 
        """
        try:
            # encoding='utf-8'은 필수입니다. (한글 깨짐 방지)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except UnicodeDecodeError:
            # 만약 utf-8로 안 열리면 cp949(윈도우 기본)로 시도
            try:
                with open(file_path, 'r', encoding='cp949') as f:
                    content = f.read()
                return content
            except Exception as e:
                print(f"❌ 텍스트 인코딩 에러: {e}")
                return ""

    def _parse_excel(self, file_path):
        """
        [업그레이드됨] 엑셀 파싱: NaN 제거 및 ' | ' 구분자로 구조 보존
        """
        try:
            # 1. 헤더 없이 읽기
            df = pd.read_excel(file_path, header=None, engine='openpyxl')

            # 2. NaN(빈 값)을 빈 문자열 ""로 치환
            df = df.fillna("")

            cleaned_text_lines = []

            # 3. 행 단위로 순회
            for index, row in df.iterrows():
                # 빈칸이 아닌 셀만 리스트로 모음
                valid_cells = [str(item).strip() for item in row if str(item).strip()]

                # 유의미한 데이터가 있는 행만 처리
                if valid_cells:
                    # 셀 사이를 " | "로 구분 (AI 힌트용)
                    row_text = " | ".join(valid_cells)
                    cleaned_text_lines.append(row_text)

            return "\n".join(cleaned_text_lines)

        except Exception as e:
            print(f"❌ 엑셀 파싱 에러: {e}")
            return ""

    def _parse_pdf(self, file_path):
        """ PDF 파싱: 텍스트 및 표 추출 """
        full_text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text: full_text += text + "\n"

                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            clean_row = [str(cell) if cell else "" for cell in row]
                            full_text += " | ".join(clean_row) + "\n"
            return full_text
        except Exception as e:
            print(f"❌ PDF 파싱 에러: {e}")
            return ""

    def _parse_word(self, file_path):
        """ Word 파싱: 문단 및 표 추출 """
        try:
            doc = docx.Document(file_path)
            full_text = []

            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text.strip())

            for table in doc.tables:
                for row in table.rows:
                    row_data = [cell.text.strip().replace('\n', ' ') for cell in row.cells]
                    if any(row_data):
                        full_text.append(" | ".join(row_data))

            return "\n".join(full_text)
        except Exception as e:
            print(f"❌ Word 파싱 에러: {e}")
            return ""


# ========================================================
# [여기가 핵심] 클래스를 밖에서 바로 쓸 수 있게 객체로 만들어둠
# ========================================================
parsing_manager = ParsingService()