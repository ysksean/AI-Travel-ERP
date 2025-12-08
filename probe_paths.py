import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sibling_travel = os.path.join(parent_dir, 'travel')

print(f"Current: {current_dir}")
print(f"Parent: {parent_dir}")
print(f"Sibling 'travel': {sibling_travel}")

if os.path.exists(sibling_travel):
    print(f"Sibling 'travel' exists.")
    try:
        items = os.listdir(sibling_travel)
        print(f"Contents of {sibling_travel}: {items}")
        
        if 'src' in items:
            src_path = os.path.join(sibling_travel, 'src')
            print(f"Found 'src': {src_path}")
            try:
                src_items = os.listdir(src_path)
                print(f"Contents of 'src': {src_items}")
                if 'services' in src_items:
                    print("Found 'services' in 'src'!")
            except Exception as e:
                print(f"Error reading src: {e}")
        
        # Also search for 'services' anywhere
        for root, dirs, files in os.walk(sibling_travel):
            if 'services' in dirs:
                print(f"Found 'services' at: {os.path.join(root, 'services')}")
    except Exception as e:
        print(f"Error reading sibling: {e}")
else:
    print("Sibling 'travel' does NOT exist.")
