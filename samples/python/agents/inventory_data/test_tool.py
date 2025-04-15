from tools import InventorySearchTool

# ツールのインスタンス化
tool = InventorySearchTool()

# 存在するIDでテスト
result = tool.run("I001")
print(f"I001のテスト結果: {result}")

# 存在しないIDでテスト
result = tool.run("I999")
print(f"I999のテスト結果: {result}") 