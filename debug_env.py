import os

filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
print(f"Checking .env at: {filepath}")

if os.path.exists(filepath):
    print(".env exists.")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            # print(f"Raw content: {content}") # Security risk, avoid printing raw
            lines = content.splitlines()
            print(f"Read {len(lines)} lines.")
            for i, line in enumerate(lines):
                line = line.strip()
                if not line or line.startswith('#'):
                    print(f"Line {i}: Skipped (empty or comment)")
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    print(f"Line {i}: Found key '{key.strip()}', Value length: {len(value.strip())}")
                else:
                    print(f"Line {i}: No '=' found: '{line}'")
    except Exception as e:
        print(f"Error reading .env: {e}")
else:
    print(".env NOT found.")
