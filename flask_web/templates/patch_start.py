import os
import sys

file_path = r'c:\AIDC\travel\flask_web\templates\product_create.html'
new_func_path = r'c:\AIDC\travel\flask_web\templates\new_start_func.js'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 0-indexed
# Line 874 -> index 873
# Line 1041 -> index 1040 (approx, need to find closing brace)
start_idx = 873

# Find end index dynamically
end_idx = -1
brace_count = 0
found_start = False

for i, line in enumerate(lines):
    if i < start_idx:
        continue
    
    if "window.startAIAnalysis = async function() {" in line:
        found_start = True
        brace_count += line.count('{')
        brace_count -= line.count('}')
        continue
    
    if found_start:
        brace_count += line.count('{')
        brace_count -= line.count('}')
        if brace_count == 0:
            end_idx = i
            break

if not found_start:
    print("Error: Start line not found.")
    sys.exit(1)

if end_idx == -1:
    print("Error: End line not found.")
    sys.exit(1)

print(f"Replacing lines {start_idx+1} to {end_idx+1}")

with open(new_func_path, 'r', encoding='utf-8') as f:
    new_func = f.read()

# Replace
lines[start_idx : end_idx + 1] = [new_func + '\n']

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Successfully patched startAIAnalysis.")
