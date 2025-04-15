# A2A連携コア機能 プロトタイプ

このリポジトリは、A2A (Agent-to-Agent) プロトコルに基づいたエージェント間連携のコア機能を検証するためのプロトタイプ実装です。

## 概要

プロトタイプは以下の2つのエージェントで構成されます。

1.  **PurchasingDataAgent (サーバーサイド)**:
    *   役割: 購買データに関する問い合わせに応答するA2Aサーバー。
    *   実装: FastAPI, LangChain (Agent, Tool), A2A common library
    *   機能: A2Aタスク (非同期) の受付、ダミーCSVデータからの情報検索 (LangChain Tool経由)、LLM連携による応答生成。
2.  **KakakuIjouSignalAgent (クライアントサイド)**:
    *   役割: 購買データの価格異常を検知する兆候エージェント。
    *   実装: LangGraph, A2A common library
    *   機能: `PurchasingDataAgent` にデータ取得を依頼 (A2A Task)、結果をポーリングで取得、取得データに基づいてLLMで異常評価、LangGraphによる状態遷移管理。

詳細は `プロトタイプ.yaml` を参照してください。

## セットアップ

### 1. 依存関係のインストール

仮想環境を作成し、必要なライブラリをインストールします。

```bash
# 仮想環境を作成 (例: venv)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

# 依存関係をインストール
pip install -r requirements.txt
```

### 2. 環境変数の設定

プロジェクトルートに `.env` ファイルを作成し、以下の環境変数を設定します。

```dotenv
# OpenAI APIキー (または使用するLLMのAPIキー)
OPENAI_API_KEY="your_openai_api_key_here"

# (任意) PurchasingDataAgentのURL (デフォルトは http://localhost:8001/a2a)
# PURCHASING_AGENT_URL="http://localhost:8001/a2a"
```

**注意:** Google Gemini など他のLLMを使用する場合は、コード (`agent.py` など) と `requirements.txt` を適宜修正し、対応するAPIキーを設定してください。

## 実行方法

### 1. PurchasingDataAgent (サーバー) の起動

ターミナル1を開き、以下のコマンドを実行します。

```bash
python -m samples.python.agents.purchasing_data
```

サーバーが `http://0.0.0.0:8001` (デフォルト) で起動し、A2Aリクエストを待ち受けます。
Agent Card は `http://localhost:8001/.well-known/agent.json` で確認できます。

### 2. KakakuIjouSignalAgent (クライアント) の実行

ターミナル2を開き、**プロジェクトルート (`A2A/A2A`) から**以下のコマンドを実行します。

**特定の製品IDで実行:**

```bash
# 例: 製品ID P001 のデータを取得・評価
python -m samples.python.agents.kakaku_ijou_signal P001

# 例: 製品ID P003 のデータを取得・評価
python -m samples.python.agents.kakaku_ijou_signal P003

# 引数なしの場合、デフォルトで P002 が実行される
python -m samples.python.agents.kakaku_ijou_signal
```

**テストケースの実行:**

```bash
# 定義済みのテストケース (P001 と P999) を実行
python -m samples.python.agents.kakaku_ijou_signal --test-all
```

クライアント側のターミナルに、LangGraphの各ノードの実行ログ、A2A通信の状況、最終的な評価結果などが表示されます。

## 検証ポイント

*   A2A非同期タスク (`/tasks/send`, `/tasks/{task_id}/result`) が正しく機能するか。
*   `PurchasingDataAgent` 内でLangChain Tool (CSV検索) が呼び出されるか。
*   `KakakuIjouSignalAgent` でA2A結果を取得し、LangGraphのStateが更新されるか。
*   LangGraphのノード遷移が意図通りか。
*   各エージェントからLLM APIが呼び出され、応答が得られるか。

詳細は `プロトタイプ.yaml` の `verification_points` を参照してください。
