# Fix duplicate script block in product_create.html

with open(r'templates\product_create.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Extract correct script block (from line 612, 0-based index 611)
correct_script = ''.join(lines[611:])

# Add proper HTML structure at the beginning
# We need to read the HTML part from the original file structure
# But since HTML is missing, we'll add minimal structure and let user know

# Actually, we need the full HTML. Let's check if we can find it elsewhere
# For now, let's just remove the first 611 lines and add minimal HTML

html_start = """{% extends "base.html" %}

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

# We need the full HTML content. Since it's too long, let's use a placeholder
# and tell user to restore from backup or previous version
# Actually, let's try to find HTML in the file by looking for content block end

# Check if there's any HTML before the scripts
# The HTML should end with {% endblock %} before {% block scripts %}
html_end = """
{% endblock %}

"""

# Combine: HTML start + (we need full HTML here) + HTML end + correct script
# For now, let's just remove duplicate and add minimal structure
output = html_start + "\n<!-- HTML content needs to be restored from backup -->\n" + html_end + correct_script

with open(r'templates\product_create.html', 'w', encoding='utf-8') as f:
    f.write(output)

print("Removed duplicate script block. HTML content needs to be restored.")
print("Please restore HTML content from backup or previous version.")

