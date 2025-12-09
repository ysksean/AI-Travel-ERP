import os
import sys

file_path = r'c:\AIDC\travel\flask_web\templates\product_create.html'
new_func_path = r'c:\AIDC\travel\flask_web\templates\restored_footer.js'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 0-indexed
# We want to replace from line 1100 (index 1099) to the end of the file.
# But wait, line 1100 was: "    // 3. UI 유틸리티 (기존 기능 유지)"
# Let's verify that line exists around there.

start_idx = 1099
found = False
for i in range(start_idx - 50, start_idx + 50):
    if i < len(lines) and "3. UI 유틸리티" in lines[i]:
        start_idx = i
        found = True
        break

if not found:
    print("Error: Start line '3. UI 유틸리티' not found.")
    # Fallback: search for dummyDB definition start if UI utility comment is missing/moved
    for i in range(len(lines)):
        if "const dummyDB = {" in lines[i]:
            start_idx = i - 4 # approximate
            found = True
            break

if not found:
    print("Critical Error: Could not find start point.")
    sys.exit(1)

print(f"Replacing from line {start_idx+1} to end.")

with open(new_func_path, 'r', encoding='utf-8') as f:
    new_footer = f.read()

# Replace from start_idx to the end
lines[start_idx:] = [new_footer]

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Successfully patched footer.")
