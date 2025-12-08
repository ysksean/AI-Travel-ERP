import os

file_path = r'c:\AIDC\travel\flask_web\templates\product_create.html'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the Top Cards section with the fixed version
# Changes:
# 1. Left Card: Removed onchange="handleQuotationFile(this)". Wrapped label in div.relative.z-20.
# 2. Right Card: Wrapped label in div.relative.z-20.
# 3. Decoration divs: Added z-0 just in case.

start_marker = '<!-- Smart Quotation & Land Operator Upload -->'
end_marker = '<form id="productForm" class="space-y-12">'

new_top_section = """<!-- Smart Quotation & Land Operator Upload -->
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
        <!-- AI Auto Input -->
        <div class="bg-gradient-to-r from-indigo-50 to-blue-50 border border-indigo-100 rounded-2xl p-8 flex flex-col items-center text-center shadow-sm relative overflow-hidden group h-full">
            <div class="absolute right-0 top-0 h-full w-1/3 bg-gradient-to-l from-indigo-100/50 to-transparent skew-x-12 pointer-events-none z-0"></div>
            <div class="w-16 h-16 bg-white rounded-2xl flex items-center justify-center shadow-sm text-indigo-600 mb-4 z-10 relative">
                <i data-lucide="sparkles" size="32"></i>
            </div>
            <h3 class="text-xl font-bold text-slate-800 z-10 relative">AI 상품 자동 생성</h3>
            <p class="text-slate-600 mt-2 mb-6 text-sm z-10 relative">견적서(Excel, Word, PDF)를 업로드하면<br>AI가 상품 정보를 자동으로 완성합니다.</p>
            <div class="w-full z-20 relative">
                <label class="w-full py-3 bg-white text-indigo-600 font-bold rounded-xl shadow-sm border border-indigo-100 hover:bg-indigo-50 cursor-pointer transition-all flex items-center justify-center gap-2">
                    <i data-lucide="upload-cloud" size="20"></i>
                    <span>파일 업로드</span>
                    <input type="file" class="hidden" accept=".xlsx,.xls,.pdf,.doc,.docx" id="topQuotationInput">
                </label>
            </div>
        </div>

        <!-- Land Operator File Upload -->
        <div class="bg-gradient-to-r from-emerald-50 to-green-50 border border-emerald-100 rounded-2xl p-8 flex flex-col items-center text-center shadow-sm relative overflow-hidden group h-full">
            <div class="absolute right-0 top-0 h-full w-1/3 bg-gradient-to-l from-emerald-100/50 to-transparent skew-x-12 pointer-events-none z-0"></div>
            <div class="w-16 h-16 bg-white rounded-2xl flex items-center justify-center shadow-sm text-emerald-600 mb-4 z-10 relative">
                <i data-lucide="file-spreadsheet" size="32"></i>
            </div>
            <h3 class="text-xl font-bold text-slate-800 z-10 relative">랜드사 가격표 분석</h3>
            <p class="text-slate-600 mt-2 mb-6 text-sm z-10 relative">가격표를 업로드하면 날짜별, 인원별 요금을<br>자동으로 추출하여 매핑합니다.</p>
            <div class="w-full z-20 relative">
                 <label class="w-full py-3 bg-white text-emerald-600 font-bold rounded-xl shadow-sm border border-emerald-100 hover:bg-emerald-50 cursor-pointer transition-all flex items-center justify-center gap-2">
                    <i data-lucide="upload-cloud" size="20"></i>
                    <span>파일 업로드</span>
                    <input type="file" name="land_itinerary_file" class="hidden" id="landFile" accept=".xlsx,.xls,.pdf,.doc,.docx" onchange="handleLandFile(this)">
                </label>
                <!-- File Info (Hidden by default) -->
                <div id="landFileInfo" class="hidden flex items-center justify-center gap-2 mt-2">
                    <span class="text-sm font-bold text-slate-800" id="landFileName">file.xlsx</span>
                    <button type="button" class="text-slate-400 hover:text-red-500" onclick="clearLandFile()"><i data-lucide="x" size="14"></i></button>
                </div>
            </div>
        </div>
    </div>

    """

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + new_top_section + content[end_idx:]
    print("Top section updated with click fixes and reverted action")
else:
    print("Top section markers not found")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
