# データクエリエージェント (Data Query Agent)

データソースに対して自然言語で問い合わせができる汎用的なQueryAgentです。設定ファイルで様々なデータソースに対応し、A2Aプロトコルでアクセス可能なエージェントとして動作します。

## 機能

- 自然言語による柔軟なデータクエリ
- CSVやExcelファイルのデータ読み込み
- 設定ファイルによるカスタマイズ
- A2Aプロトコル対応エージェントとして実行可能
- 対話モードでのテスト実行

## セットアップ

### 前提条件

- Python 3.8以上
- 必要パッケージ:
  - pandas
  - langchain-openai
  - pyyaml
  - numpy
  - uvicorn
  - fastapi (A2A Serverの依存関係)

### インストール

```bash
# 必要なパッケージをインストール
pip install pandas langchain-openai pyyaml numpy uvicorn fastapi
```

## 使用方法

### 1. 対話モードでのテスト実行

QueryAgentの機能を対話的にテストするには、以下のコマンドを実行します：

```powershell
# まず仮想環境を有効化（必要に応じて）
.\venv\Scripts\Activate

# data_agentの親ディレクトリに移動
cd A2A_risk/samples/python/agents

# パッケージとしてテストを実行
python -m A2A_risk.samples.python.agents.data_agent.tests.test_run_agent purchase_orders_config.yaml
python -m A2A_risk.samples.python.agents.data_agent.tests.test_run_agent 不正事例.yaml
```

- 設定ファイル（YAML）は `data_agent/config/` ディレクトリ内のファイル名を指定してください。
- 例: `purchase_orders_config.yaml` など。
- 実行時のカレントディレクトリが `data_agent` の親ディレクトリであることを確認してください。

「exit」または「quit」と入力することで終了できます。

### 2. A2Aエージェントとして起動

A2Aプロトコルに対応したエージェントとして起動するには、まずA2A_riskディレクトリに移動してから、以下のコマンドを実行します：

```powershell
cd A2A_risk
python -m samples.python.agents.data_agent --config samples/python/agents/data_agent/config/purchase_orders_config.yaml
```

ポートを変更する場合：

```powershell
python -m samples.python.agents.data_agent --config samples/python/agents/data_agent/config/purchase_orders_config.yaml --port 8123
```

### 3. 複数のエージェントを一括起動する方法

複数のデータエージェントを一度に起動するには、用意されたPowerShellスクリプトを使用します：

```powershell
# 任意の場所からスクリプトを実行する場合
& "C:\Users\nyham\work\A2A\A2A\samples\python\agents\data_agent\start_all_agents.ps1"

# またはA2Aプロジェクトのルートディレクトリから実行する場合
cd C:\Users\nyham\work\A2A\A2A
.\samples\python\agents\data_agent\start_all_agents.ps1
```

このスクリプトは、各設定ファイルごとに別々のPowerShellウィンドウを開き、それぞれのデータエージェントを起動します。ウィンドウは手動で閉じるまで開いたままになります。

> **注意**: スクリプトの実行にはPowerShellの実行ポリシーが関係する場合があります。必要に応じて次のコマンドで実行を許可してください：
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
> ```

### 4. エージェントカードの確認方法

A2Aエージェントが起動したら、ブラウザまたはcURLなどで以下のURLにアクセスすることでエージェントカードを確認できます：

```
http://localhost:8001/.well-known/agent.json
```

※ポート番号を変更した場合は、そのポート番号に置き換えてください。

### 5. A2Aエージェントへのアクセス

A2Aエージェントが起動したら、A2Aプロトコルに対応したクライアントや以下のようなcURLコマンドでアクセスできます：

```bash
curl -X POST http://localhost:8001/a2a -H "Content-Type: application/json" -d '{
  "jsonrpc": "2.0",
  "method": "sendTask",
  "id": "1",
  "params": {
    "id": "task-001",
    "message": {
      "role": "user",
      "parts": [
        {
          "text": "商品の平均価格はいくらですか？"
        }
      ]
    },
    "historyLength": 10
  }
}'
```

## 設定ファイル

エージェントの設定は、YAMLファイルで管理します。以下は設定ファイルの例です：

```yaml
# 基本設定
agent_name: "データクエリエージェント"
agent_description: "データソースに対して自然言語でクエリができます"
host: "0.0.0.0"
port: 8001
llm_model: "gpt-4o-mini"
data_source: "../data/sample_data.csv"

# A2A設定
organization: "あなたの組織名"
version: "1.0.0"
endpoint: "/a2a"

# UI/UX設定
defaultInputModes: ["text"]
defaultOutputModes: ["text"]

# スキル定義
skills:
  - id: "query_data"
    name: "データクエリ"
    description: "データに対して自然言語で問い合わせができます"
    examples:
      - "商品の平均価格はいくらですか？"
      - "最も在庫数の多い商品は何ですか？"
      - "カテゴリAの商品すべてを教えてください"
    inputModes: ["text"]
    outputModes: ["text"]
```

### 設定項目の説明

| 項目 | 説明 | デフォルト値 |
|------|------|------------|
| agent_name | エージェントの名前 | "Generic Data Agent" |
| agent_description | エージェントの説明 | "Analyzes data based on configuration." |
| host | サーバーのホスト | "0.0.0.0" |
| port | サーバーのポート | 8001 |
| llm_model | 使用するLLMモデル | "gpt-4o-mini" |
| data_source | データソースのパス (必須) | なし |
| organization | 提供組織名 | "Your Organization" |
| version | エージェントのバージョン | "1.0.0" |
| endpoint | A2Aエンドポイント | "/a2a" |
| defaultInputModes | 入力モードのリスト | ["text"] |
| defaultOutputModes | 出力モードのリスト | ["text"] |
| skills | スキル定義のリスト | (デフォルトスキル) |

## 新規エージェントの作成方法

### 1. データの準備

まずデータファイルを準備します。現在サポートされているのは以下の形式です：
- CSV (.csv)
- Excel (.xlsx, .xls)

例えば、CSVデータファイルを以下のように作成します：
```csv
id,name,category,price,stock
P001,ノートパソコン,電子機器,120000,5
P002,デスクトップPC,電子機器,150000,3
P003,ワイヤレスマウス,アクセサリ,3500,20
```

このファイルを `samples/python/agents/data_agent/data/` ディレクトリに配置します。

### 2. 設定ファイルの作成

次に、エージェント用の設定ファイルを作成します。
`samples/python/agents/data_agent/config/` ディレクトリに、例えば `my_agent_config.yaml` という名前で設定ファイルを作成します：

```yaml
agent_name: "私のデータエージェント"
agent_description: "自社製品データを検索・分析するエージェント"
host: "0.0.0.0"
port: 8002
llm_model: "gpt-4o-mini"
data_source: "../data/my_product_data.csv"
organization: "私の会社"
version: "1.0.0"

# スキル定義
skills:
  - id: "product_search"
    name: "製品検索"
    description: "製品情報を検索します"
    examples:
      - "製品Xの価格はいくらですか？"
      - "カテゴリYの製品をすべて表示してください"
    inputModes: ["text"]
    outputModes: ["text"]
```

### 3. エージェントの実行

作成した設定ファイルを使ってエージェントを起動します：

```bash
python -m samples.python.agents.data_agent --config samples/python/agents/data_agent/config/my_agent_config.yaml
```

これで新しいエージェントが起動し、指定したデータソースに対して質問ができるようになります。

## トラブルシューティング

### データ読み込みエラー

- データファイルのパスが正しいか確認してください。相対パスは設定ファイルからの相対パスとして解釈されます。
- 設定ファイル名のスペルミス（例: `purchasing_config.yaml` ではなく `purchase_orders_config.yaml` など）に注意してください。

### LLMモデルエラー

OpenAI APIキーが環境変数 `OPENAI_API_KEY` に設定されていることを確認してください。

### A2A通信エラー

- ポートが他のアプリケーションで使用されていないか確認
- ファイアウォール設定でポートが開放されているか確認

## 拡張と貢献

このプロジェクトは以下の方法で拡張可能です：

- 新しいデータソース形式のサポート追加
- UIの改善
- パフォーマンス最適化

## ライセンス

このプロジェクトは [ライセンス情報] の下で公開されています。

---

作成者: [作成者情報]
連絡先: [連絡先情報] 