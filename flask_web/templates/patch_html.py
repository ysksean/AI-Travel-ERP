import os
import sys

file_path = r'c:\AIDC\travel\flask_web\templates\product_create.html'
new_part_path = r'c:\AIDC\travel\flask_web\templates\restored_html_part.html'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 0-indexed
# Line 74 in 1-indexed is index 73
target_idx = 73

if target_idx >= len(lines):
    print("Error: Target index out of range.")
    sys.exit(1)

# Verify the broken line
if "<label <form" not in lines[target_idx]:
    print(f"Error: Target line mismatch. Found: {lines[target_idx]}")
    # Try to find it nearby
    found = False
    for i in range(target_idx - 5, target_idx + 5):
        if i < len(lines) and "<label <form" in lines[i]:
            target_idx = i
            found = True
            break
    if not found:
        print("Critical Error: Could not find broken line.")
        sys.exit(1)

print(f"Replacing line {target_idx+1}")

with open(new_part_path, 'r', encoding='utf-8') as f:
    new_part = f.read()

# Replace the single broken line with the new multi-line content
lines[target_idx] = new_part + '\n'

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Successfully patched HTML.")
