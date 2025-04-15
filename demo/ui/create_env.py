# create_env.py
import os

api_key = "AIzaSyAGqg3Whxe8dLPX-qTB5H1YOE2NvuaUMP4" # <-- ここに実際のAPIキーを貼り付け（"" は不要）
env_content = f"GOOGLE_API_KEY={api_key}"
env_path = os.path.join(os.path.dirname(__file__), '.env')

try:
    with open(env_path, 'w', encoding='utf-8') as f:
        f.write(env_content)
    print(f"'{env_path}' を UTF-8 (BOMなし) で作成しました。")
    print(f"書き込まれた内容: {env_content}") # 確認用
except Exception as e:
    print(f"ファイル作成中にエラーが発生しました: {e}") 