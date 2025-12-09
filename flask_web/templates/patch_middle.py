import os
import sys

file_path = r'c:\AIDC\travel\flask_web\templates\product_create.html'
new_middle_path = r'c:\AIDC\travel\flask_web\templates\restored_middle.js'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# We need to find the start and end points.
# Start: "} catch (er" (approx line 1052 in view)
# End: "window.changeMonth = function (delta) {" (approx line 1119 in view)
# Wait, the footer patch started at "3. UI 유틸리티" which was around line 1125.
# So we should replace up to that point.

start_idx = -1
end_idx = -1

# Find start
for i in range(1000, 1100):
    if i < len(lines) and "} catch (er" in lines[i]:
        start_idx = i
        break

if start_idx == -1:
    print("Error: Start point '} catch (er' not found.")
    # Fallback search
    for i in range(1000, 1100):
        if "console.error(error);" in lines[i]:
            start_idx = i - 1
            break

if start_idx == -1:
    print("Critical Error: Could not find start point.")
    sys.exit(1)

# Find end
# The footer patch replaced from "3. UI 유틸리티" onwards.
# So we should stop right before that.
for i in range(start_idx, len(lines)):
    if "3. UI 유틸리티" in lines[i]:
        end_idx = i
        break

if end_idx == -1:
    print("Error: End point '3. UI 유틸리티' not found.")
    # Fallback: look for dummyDB
    for i in range(start_idx, len(lines)):
        if "const dummyDB = {" in lines[i]:
            end_idx = i - 4
            break

if end_idx == -1:
    print("Critical Error: Could not find end point.")
    sys.exit(1)

print(f"Replacing from line {start_idx+1} to {end_idx}")

with open(new_middle_path, 'r', encoding='utf-8') as f:
    new_middle = f.read()

lines[start_idx:end_idx] = [new_middle + '\n']

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Successfully patched middle section.")
