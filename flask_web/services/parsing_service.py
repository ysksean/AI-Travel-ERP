import os
import pandas as pd
import re
import json
import pdfplumber
import docx
from docx.document import Document


class ParsingService:
    def parse_file(self, file_path):
        """
        파일을 읽어서 LLM이 이해하기 좋은 'Markdown 포맷'으로 전처리하여 반환합니다.
        (AI 프롬프트 생성 단계 제외)
        """
        if not os.path.exists(file_path):
            return {"error": "File not found"}

        ext = file_path.split('.')[-1].lower()
        raw_text_data = ""

        try:
            if ext in ['xlsx', 'xls']:
                raw_text_data = self._extract_excel_to_markdown(file_path)
            elif ext == 'pdf':
                raw_text_data = self._extract_pdf_to_markdown(file_path)
            elif ext == 'docx':
                raw_text_data = self._extract_word_to_markdown(file_path)
            elif ext == 'txt':
                raw_text_data = self._extract_text_pure(file_path)
            else:
                return {"error": "Unsupported format"}

            # 전처리된 텍스트(Markdown)를 바로 반환
            return raw_text_data

        except Exception as e:
            return {"error": str(e)}

    # ---------------------------------------------------------
    # [전처리 엔진] 각 파일을 LLM이 좋아하는 'Markdown' 형태로 변환
    # ---------------------------------------------------------

    def _extract_excel_to_markdown(self, file_path):
        """
        [최신 기법] 엑셀을 마크다운 표 형태로 변환
        LLM은 '| 컬럼 | 값 |' 형태를 표로 완벽하게 인식합니다.
        """
        text_buffer = []
        # openpyxl 엔진 사용, data_only=True로 수식 대신 값만 가져옴
        # header=None: 첫 줄을 헤더로 잡지 않고 데이터로 처리 (Unnamed 방지)
        xls = pd.read_excel(file_path, sheet_name=None, engine='openpyxl', header=None)

        for sheet_name, df in xls.items():
            text_buffer.append(f"\n## Sheet: {sheet_name}\n")

            # 데이터가 없는 경우 스킵
            if df.empty:
                continue

            # 헤더가 엉망인 경우를 대비해 헤더를 없애고 모든 데이터를 내용으로 처리할 수도 있으나,
            # 여기서는 결측치(NaN)를 빈칸으로 채우고 마크다운으로 변환
            df = df.fillna("")

            # DataFrame을 Markdown Table로 변환 (tabulate가 없어도 pipe format 사용)
            # to_markdown()을 쓰려면 tabulate 라이브러리가 필요하므로, 없으면 수동 변환
            try:
                # headers=[] 옵션으로 숫자 헤더(0, 1, 2...) 출력 방지
                markdown_table = df.to_markdown(index=False, headers=[])
            except (ImportError, TypeError):
                # tabulate 라이브러리가 없는 환경을 위한 수동 변환
                # header=False로 숫자 헤더 출력 방지
                markdown_table = df.to_csv(sep="|", index=False, header=False)

            text_buffer.append(markdown_table)
            text_buffer.append("\n---\n")

        return self._clean_text("\n".join(text_buffer))

    def _extract_pdf_to_markdown(self, file_path):
        """
        [최신 기법] PDF의 텍스트와 표를 분리하여 마크다운으로 재조립
        """
        text_buffer = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text_buffer.append(f"\n### Page {i + 1}\n")

                # 1. 텍스트 추출
                text = page.extract_text()
                if text:
                    text_buffer.append(text)

                # 2. 표 추출 및 마크다운 변환
                tables = page.extract_tables()
                for table in tables:
                    if not table: continue

                    # 표 시작 알림
                    text_buffer.append("\n[Table Data]:")

                    # 리스트 형태의 표를 마크다운 문자열로 변환
                    # 예: [['이름', '나이'], ['홍길동', '20']] -> | 이름 | 나이 |\n|---|---|\n| 홍길동 | 20 |
                    headers = table[0]
                    rows = table[1:]

                    # 헤더 처리
                    header_str = "| " + " | ".join([str(h).replace('\n', ' ') if h else '' for h in headers]) + " |"
                    separator = "| " + " | ".join(['---'] * len(headers)) + " |"
                    text_buffer.append(header_str)
                    text_buffer.append(separator)

                    # 행 처리
                    for row in rows:
                        row_str = "| " + " | ".join([str(c).replace('\n', ' ') if c else '' for c in row]) + " |"
                        text_buffer.append(row_str)
                    text_buffer.append("\n")

        return self._clean_text("\n".join(text_buffer))

    def _extract_word_to_markdown(self, file_path):
        """ Word 문서를 마크다운 구조로 변환 """
        doc = docx.Document(file_path)
        text_buffer = []

        for element in doc.element.body:
            if isinstance(element, docx.oxml.text.paragraph.CT_P):
                para = docx.text.paragraph.Paragraph(element, doc)
                if para.text.strip():
                    text_buffer.append(para.text)
            elif isinstance(element, docx.oxml.table.CT_Tbl):
                table = docx.table.Table(element, doc)
                text_buffer.append("\n[Table Data]:")

                rows_data = []
                for row in table.rows:
                    row_cells = [cell.text.strip().replace('\n', ' ') for cell in row.cells]
                    rows_data.append("| " + " | ".join(row_cells) + " |")

                if rows_data:
                    # 헤더 구분선 추가 (첫 줄을 헤더로 가정)
                    text_buffer.append(rows_data[0])
                    if len(rows_data) > 1:
                        col_count = rows_data[0].count('|') - 1
                        text_buffer.append("| " + " | ".join(['---'] * col_count) + " |")
                        text_buffer.extend(rows_data[1:])
                text_buffer.append("\n")

        return self._clean_text("\n".join(text_buffer))

    def _extract_text_pure(self, file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return self._clean_text(f.read())

    def _clean_text(self, text):
        """
        [전처리 핵심] LLM 토큰 절약을 위한 텍스트 클리닝
        """
        # 1. 연속된 공백 제거
        text = re.sub(r' +', ' ', text)
        # 2. 연속된 줄바꿈을 최대 2개로 제한 (문단 구분용)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 3. 탭 문자 제거
        text = text.replace('\t', ' ')
        return text.strip()

    # ---------------------------------------------------------
    # [호환성 메서드] 구버전 메서드명을 사용하는 호출을 위한 연결 (Alias)
    # 기존 코드에서 _parse_excel 등을 호출해도 문제없도록 처리
    # ---------------------------------------------------------
    def _parse_excel(self, file_path):
        return self._extract_excel_to_markdown(file_path)

    def _parse_pdf(self, file_path):
        return self._extract_pdf_to_markdown(file_path)

    def _parse_word(self, file_path):
        return self._extract_word_to_markdown(file_path)

    def _parse_txt(self, file_path):
        return self._extract_text_pure(file_path)


# ========================================================
# 사용 예시
# ========================================================
parsing_manager = ParsingService()