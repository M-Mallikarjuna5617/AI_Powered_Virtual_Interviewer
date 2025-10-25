import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check keys
openai_key = os.getenv('OPENAI_API_KEY')
judge0_key = os.getenv('JUDGE0_API_KEY')

print("="*50)
print("API KEYS CHECK")
print("="*50)

if openai_key:
    print(f"✅ OpenAI Key Found: {openai_key[:15]}...")
    if openai_key.startswith('sk-'):
        print("✅ OpenAI Key Format: Correct")
    else:
        print("❌ OpenAI Key Format: Wrong (should start with sk-)")
else:
    print("❌ OpenAI Key: NOT FOUND")

print()

if judge0_key:
    print(f"✅ Judge0 Key Found: {judge0_key[:15]}...")
    if len(judge0_key) >= 40:
        print("✅ Judge0 Key Length: Correct")
    else:
        print("❌ Judge0 Key Length: Too short")
else:
    print("❌ Judge0 Key: NOT FOUND")

print("="*50)