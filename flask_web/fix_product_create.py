# Fix product_create.html - Remove duplicate script block and add proper HTML structure

with open(r'templates\product_create.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 612줄부터 끝까지가 올바른 스크립트 블록 (0-based index: 611)
correct_script = ''.join(lines[611:])

# HTML 부분 읽기 (이전에 저장했던 clean 파일이 있다면)
# 없으면 직접 만들어야 함
html_content = """{% extends "base.html" %}

{% block title %}상품 등록{% endblock %}

{% block content %}
<div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
    <div class="mb-8 flex items-center justify-between">
        <div>
            <h1 class="text-2xl font-bold text-slate-800">상품 등록</h1>
            <p class="text-slate-500 mt-2 text-lg">여행의 가치를 높이는 매력적인 상품을 설계하세요.</p>
        </div>
        <button type="button" onclick="startAIAnalysis()"
            class="px-6 py-3 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl font-bold shadow-lg hover:shadow-xl transition-all flex items-center gap-2">
            <i data-lucide="sparkles" size="20"></i>
            <span>AI 분석 시작</span>
        </button>
    </div>
"""

# 실제 HTML 부분을 찾기 위해 원본 백업 확인
# 일단 스크립트 블록만 올바르게 유지하고 HTML은 나중에
# 612줄 이전에 HTML이 있어야 하는데, 현재는 스크립트만 있음

# 간단한 해결: 첫 611줄 제거하고 올바른 HTML + 스크립트 조합
# 하지만 HTML 부분이 없으므로, 원본 파일에서 HTML 부분을 찾아야 함

# 일단 올바른 스크립트만 남기고 HTML 부분은 나중에 추가
# 스크립트 블록이 올바르게 시작하는지 확인
if correct_script.startswith('{% block scripts %}'):
    print("Found correct script block starting at line 612")
    # HTML 부분을 찾기 위해 다른 템플릿 참조
    # 일단 스크립트만 저장
    with open(r'templates\product_create_script_only.html', 'w', encoding='utf-8') as f:
        f.write(correct_script)
    print("Script block saved to product_create_script_only.html")
else:
    print("Error: Script block not found at expected location")

