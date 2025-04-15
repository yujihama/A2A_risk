# test_dotenv.py
import os
from dotenv import load_dotenv

print("テストスクリプト開始")
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
print(f".env ファイルのパス: {dotenv_path}")

# verbose=True と override=True をつけて実行してみる
success = load_dotenv(dotenv_path=dotenv_path, verbose=True, override=True)
print(f"load_dotenv の成功ステータス: {success}") # True が返るはず

api_key = os.getenv('GOOGLE_API_KEY')
print(f"読み込み後の GOOGLE_API_KEY: {api_key}")

if api_key:
    print(">>> 成功: GOOGLE_API_KEY が読み込めました。")
else:
    print(">>> 失敗: GOOGLE_API_KEY が読み込めませんでした。")

print("テストスクリプト終了") 