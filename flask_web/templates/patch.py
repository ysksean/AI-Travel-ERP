import os
import sys

file_path = r'c:\AIDC\travel\flask_web\templates\product_create.html'
new_func_path = r'c:\AIDC\travel\flask_web\templates\new_func.js'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 0-indexed
# Line 874 -> index 873
# Line 1041 -> index 1040
start_idx = 873
end_idx = 1040

# Verify start line
if "window.startAIAnalysis = async function() {" not in lines[start_idx]:
    print(f"Error: Start line mismatch. Found: {lines[start_idx]}")
    sys.exit(1)

# Verify end line
if "}" not in lines[end_idx]:
    print(f"Error: End line mismatch. Found: {lines[end_idx]}")
    sys.exit(1)

with open(new_func_path, 'r', encoding='utf-8') as f:
    new_func = f.read()

# Replace
# We replace lines[start_idx : end_idx + 1] with the new function
lines[start_idx : end_idx + 1] = [new_func + '\n']

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Successfully patched file.")
