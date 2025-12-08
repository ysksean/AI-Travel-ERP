import os

file_path = r'c:\AIDC\travel\flask_web\templates\product_create.html'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Replace Top Cards Section
# I'll look for the start of the grid and replace the whole block to ensure consistency.
start_marker = '<!-- Smart Quotation & Land Operator Upload -->'
end_marker = '<form id="productForm" class="space-y-12">'

new_top_section = """<!-- Smart Quotation & Land Operator Upload -->
    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
        <!-- AI Auto Input -->
        <div class="bg-gradient-to-r from-indigo-50 to-blue-50 border border-indigo-100 rounded-2xl p-8 flex flex-col items-center text-center shadow-sm relative overflow-hidden group h-full">
            <div class="w-16 h-16 bg-white rounded-2xl flex items-center justify-center shadow-sm text-indigo-600 mb-4 z-10">
                <i data-lucide="sparkles" size="32"></i>
            </div>
            <h3 class="text-xl font-bold text-slate-800 z-10">AI 상품 자동 생성</h3>
            <p class="text-slate-600 mt-2 mb-6 text-sm z-10">견적서(Excel, Word, PDF)를 업로드하면<br>AI가 상품 정보를 자동으로 완성합니다.</p>
            <label class="w-full py-3 bg-white text-indigo-600 font-bold rounded-xl shadow-sm border border-indigo-100 hover:bg-indigo-50 cursor-pointer transition-all flex items-center justify-center gap-2 z-10 relative">
                <i data-lucide="upload-cloud" size="20"></i>
                <span>파일 업로드</span>
                <input type="file" class="hidden" accept=".xlsx,.xls,.pdf,.doc,.docx" id="topQuotationInput" onchange="handleQuotationFile(this)">
            </label>
            <div class="absolute right-0 top-0 h-full w-1/3 bg-gradient-to-l from-indigo-100/50 to-transparent skew-x-12 pointer-events-none"></div>
        </div>

        <!-- Land Operator File Upload -->
        <div class="bg-gradient-to-r from-emerald-50 to-green-50 border border-emerald-100 rounded-2xl p-8 flex flex-col items-center text-center shadow-sm relative overflow-hidden group h-full">
            <div class="w-16 h-16 bg-white rounded-2xl flex items-center justify-center shadow-sm text-emerald-600 mb-4 z-10">
                <i data-lucide="file-spreadsheet" size="32"></i>
            </div>
            <h3 class="text-xl font-bold text-slate-800 z-10">랜드사 가격표 분석</h3>
            <p class="text-slate-600 mt-2 mb-6 text-sm z-10">가격표를 업로드하면 날짜별, 인원별 요금을<br>자동으로 추출하여 매핑합니다.</p>
            <div class="w-full z-10 relative">
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
        </div>
    </div>

    """

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + new_top_section + content[end_idx:]
    print("Top section updated")
else:
    print("Top section markers not found")

# 2. Add handleQuotationFile function
# I'll append it before `window.handleLandFile`
js_marker = 'window.handleLandFile = function (input) {'
js_code = """
        window.handleQuotationFile = async function(input) {
            const file = input.files[0];
            if (!file) return;

            // Show loading state
            const label = input.closest('label');
            const originalContent = label.innerHTML;
            // Keep input in DOM but hidden/disabled effectively
            label.classList.add('opacity-75', 'cursor-wait');
            label.innerHTML = '<i data-lucide="loader-2" class="animate-spin" size="20"></i><span>분석 중...</span>';
            lucide.createIcons();

            // Simulate processing
            await new Promise(r => setTimeout(r, 2000));

            // Trigger AI Content Generation
            await generateAIContent();

            // Reset button
            label.classList.remove('opacity-75', 'cursor-wait');
            label.innerHTML = originalContent;
            // Re-attach onchange is not needed because we replaced innerHTML but input was inside... 
            // Wait, replacing innerHTML destroys the input element if it was inside.
            // Better approach: Don't replace innerHTML, just change text and icon.
            
            // Actually, let's just reload the page or show a success message. 
            // But for this demo, let's just restore the button.
            // Since we replaced innerHTML, we lost the input. We need to re-add it or handle differently.
            // Let's just change the span text and icon.
        }
        
        // Revised function to avoid destroying input
        window.handleQuotationFile = async function(input) {
            const file = input.files[0];
            if (!file) return;
            
            const label = input.closest('label');
            const icon = label.querySelector('i');
            const span = label.querySelector('span');
            const originalIcon = icon.getAttribute('data-lucide');
            const originalText = span.textContent;
            
            icon.setAttribute('data-lucide', 'loader-2');
            icon.classList.add('animate-spin');
            span.textContent = '분석 중...';
            lucide.createIcons();
            
            await new Promise(r => setTimeout(r, 2000));
            
            await generateAIContent();
            
            icon.setAttribute('data-lucide', originalIcon);
            icon.classList.remove('animate-spin');
            span.textContent = originalText;
            lucide.createIcons();
            
            // Clear input
            input.value = '';
        }

        """

if js_marker in content:
    content = content.replace(js_marker, js_code + js_marker)
    print("JS function added")
else:
    print("JS marker not found")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
