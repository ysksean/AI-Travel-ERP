import sys
import os

# Add manual env loader logic here or import from app?
# Safest to copy paste the loader to ensure it works in isolation quickly

def load_env_manual(filepath):
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' in line:
                        key, value = line.split('=', 1)
                        value = value.strip()
                        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                            value = value[1:-1]
                        os.environ[key.strip()] = value
    except Exception:
        pass

load_env_manual(os.path.join(os.path.dirname(__file__), '.env'))

# Normalize
if not os.getenv('GOOGLE_API_KEY') and os.getenv('MY_API_KEY'):
    os.environ['GOOGLE_API_KEY'] = os.getenv('MY_API_KEY')

import google.generativeai as genai

api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    print("No API KEY found.")
    sys.exit(1)

genai.configure(api_key=api_key)

print("Listing available models...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print(f"Error listing models: {e}")
