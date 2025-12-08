import os
import re

file_path = r'c:\AIDC\travel\flask_web\templates\product_create.html'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: dummyDB split strings
# We look for lines ending with " and the next line starting with text, or split in middle of string
# The specific case observed:
# desc: "18홀 • 콜린 몽고메리
# 설계",
# We can just replace the specific broken strings we saw.

broken_string_1 = """desc: "18홀 • 콜린 몽고메리
설계","""
fixed_string_1 = """desc: "18홀 • 콜린 몽고메리 설계","""

if broken_string_1 in content:
    content = content.replace(broken_string_1, fixed_string_1)
    print("Fixed broken string 1")
else:
    # Try regex if exact match fails due to whitespace
    content = re.sub(r'desc: "18홀 • 콜린 몽고메리\s+설계",', 'desc: "18홀 • 콜린 몽고메리 설계",', content)
    print("Attempted regex fix for string 1")

# Fix 2: fillAllPolicies split string
broken_string_2 = """- 여행 개시 20일 전까지 통보 시: 여행요금의
            10 % 배상";"""
fixed_string_2 = """- 여행 개시 20일 전까지 통보 시: 여행요금의 10 % 배상";"""

if broken_string_2 in content:
    content = content.replace(broken_string_2, fixed_string_2)
    print("Fixed broken string 2")
else:
    # Try regex
    content = re.sub(r'- 여행 개시 20일 전까지 통보 시: 여행요금의\s+10 % 배상";', '- 여행 개시 20일 전까지 통보 시: 여행요금의 10 % 배상";', content)
    print("Attempted regex fix for string 2")

# Fix 3: Check for other split strings in dummyDB
# Example: desc: "5성급 • 프라이빗 풀빌라",
# The view showed:
# desc: "5성급 • 프라이빗 풀빌라",
#                 loc: "호이안, 베트남"
# This is fine because the newline is *after* the comma.

# Example: desc: "4성급 • 한강 뷰", loc:
#                     "다낭 시내"
# This is fine because newline is after `loc:`

# Example: image:
#                     "https://..."
# This is fine.

# Let's just save the fixes.

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
