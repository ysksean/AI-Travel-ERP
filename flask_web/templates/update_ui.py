import os

file_path = r'c:\AIDC\travel\flask_web\templates\product_create.html'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 2. Right Card Replacement
right_card_old = """        <!-- Land Operator File Upload -->
        <div
            class="bg-gradient-to-r from-emerald-50 to-green-50 border border-emerald-100 rounded-2xl p-8 flex items-center justify-between shadow-sm relative overflow-hidden group h-full">
            <div class="flex items-center gap-6 z-10">
                <div
                    class="w-16 h-16 bg-white rounded-2xl flex items-center justify-center shadow-sm text-emerald-600 shrink-0">
                    <i data-lucide="file-spreadsheet" size="32"></i>
                </div>
                <div class="flex-1">
                    <h3 class="text-xl font-bold text-slate-800">랜드사 가격표 분석</h3>
                    <p class="text-slate-600 mt-1 text-sm">가격표를 업로드하면 날짜별, 인원별 요금을 자동으로 추출하여 매핑합니다.</p>
                    <div class="flex items-center gap-4">
                        <label
                            class="px-6 py-3 bg-white text-emerald-600 font-bold rounded-xl shadow-sm border border-emerald-100 hover:bg-emerald-50 cursor-pointer transition-all flex items-center gap-2 whitespace-nowrap">
                            <i data-lucide="upload" size="20"></i>
                            <span>파일 선택</span>
                            <input type="file" name="land_itinerary_file" class="hidden" id="landFile"
                                accept=".xlsx,.xls,.pdf,.doc,.docx" onchange="handleLandFile(this)">
                        </label>
                        <div id="landFileInfo" class="hidden flex items-center gap-2">
                            <span class="text-sm font-bold text-slate-800" id="landFileName">file.xlsx</span>
                            <span class="px-2 py-0.5 bg-yellow-100 text-yellow-700 text-xs font-bold rounded-full">분석 대기
                                중</span>
                            <button type="button" class="text-slate-400 hover:text-red-500" onclick="clearLandFile()"><i
                                    data-lucide="x" size="14"></i></button>
                        </div>
                    </div>
                </div>
            </div>
        </div>"""

right_card_new = """        <!-- Land Operator File Upload -->
        <div class="bg-gradient-to-r from-emerald-50 to-green-50 border border-emerald-100 rounded-2xl p-8 flex flex-col items-center text-center shadow-sm relative overflow-hidden group h-full">
            <div class="w-16 h-16 bg-white rounded-2xl flex items-center justify-center shadow-sm text-emerald-600 mb-4 z-10">
                <i data-lucide="file-spreadsheet" size="32"></i>
            </div>
            <h3 class="text-xl font-bold text-slate-800 z-10">랜드사 가격표 분석</h3>
            <p class="text-slate-600 mt-2 mb-6 text-sm z-10">가격표를 업로드하면 날짜별, 인원별 요금을<br>자동으로 추출하여 매핑합니다.</p>
            <div class="w-full z-10">
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
            <div class="absolute right-0 top-0 h-full w-1/3 bg-gradient-to-l from-emerald-100/50 to-transparent skew-x-12 pointer-events-none"></div>
        </div>"""

if right_card_old in content:
    content = content.replace(right_card_old, right_card_new)
    print("Right card updated")
else:
    print("Right card not found")
    # Debug: print a small chunk to see what's there
    print("Content around Right Card:")
    start_idx = content.find("<!-- Land Operator File Upload -->")
    if start_idx != -1:
        print(content[start_idx:start_idx+200])
    else:
        print("Right card comment not found")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
