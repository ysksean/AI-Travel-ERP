import sys
import os

# Add current dir to sys.path
sys.path.append(os.getcwd())

# Import app to trigger the load_env_manual call
try:
    import app
except ImportError:
    pass # Might fail on other things but env loading happens early

key = os.environ.get('GOOGLE_API_KEY')
if key:
    print(f"GOOGLE_API_KEY found: {key[:5]}...")
else:
    print("GOOGLE_API_KEY NOT found.")
