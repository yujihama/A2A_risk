# 自律型異常検知エージェント

このモジュールは、YAMLで定義された様々なシナリオに基づいて、製品データの異常を自律的に検知するエージェントです。例えば、価格乖離、取引数量の異常、在庫状況など、シナリオに記述された観点から柔軟に分析を行います。

## 特徴

- **自律的計画立案**: LLMを活用して、シナリオと入力パラメータに基づいた実行計画を自動生成します
- **マルチステップ処理**: 計画は複数のステップに分割され、必要に応じて異なるエージェントを連携します
- **シナリオベース**: YAMLで定義された多様なシナリオを実行でき、様々な観点から異常検知が可能です
- **データ主導型分析**: 収集したデータに基づき、LLMがシナリオに従って分析を実施します
- **YAML設定**: 接続先エージェントの設定をYAMLファイルで柔軟に指定可能

## 実行方法

### 基本的な使用法

シナリオIDを指定して実行:

```
python -m samples.python.agents.smart_kakaku_signal run --scenario-id transaction_volume_anomaly --params '{"product_id": "P001"}'
```

シナリオを直接指定して実行:

```
python -m samples.python.agents.smart_kakaku_signal run --scenario-text "指定された製品IDの在庫数が5個未満の場合に異常と判定します。" --params '{"product_id": "P001"}'
```

### 設定ファイルの使用

カスタム設定ファイルを指定:

```
python -m samples.python.agents.smart_kakaku_signal --config custom_config.yaml run --scenario-id transaction_volume_anomaly
```

設定の一時的な変更:

```
python -m samples.python.agents.smart_kakaku_signal --update-config "agents.purchasing_data.url=http://localhost:5001" run --scenario-id transaction_volume_anomaly
```

### テスト実行

すべてのテストケースを実行:

```
python -m samples.python.agents.smart_kakaku_signal --test-all
```

## 設定ファイル (agent_config.yaml)

エージェントの接続先や設定は `agent_config.yaml` ファイルで管理されています:

```yaml
agents:
  # 購買データエージェント
  purchasing_data:
    name: PurchasingDataAgent
    url: http://localhost:8001
    description: 購買データを提供するエージェント
  
  # 在庫データエージェント
  inventory_data:
    name: InventoryDataAgent
    url: http://localhost:8002
    description: 在庫データを提供するエージェント
```

## 動作の流れ

1. **シナリオ解析**: LLMが指定されたシナリオを解析して、何を検知すべきかを理解します
2. **計画立案**: シナリオの内容に基づいた実行計画を自動生成します
3. **データ収集**: 計画に基づき、必要なエージェントと連携してデータを収集します
4. **分析実行**: 収集したデータをシナリオに従って分析し、異常の有無を判定します
5. **結果報告**: 分析結果と詳細な説明を表示します

## 設定

エージェントの動作には以下の環境変数が必要です:

- `OPENAI_API_KEY`: LLMを使用するためのOpenAI APIキー

## 前提条件

- PurchasingDataAgent と InventoryDataAgent が実行中であること（デフォルトのURLはそれぞれ http://localhost:8001 と http://localhost:8002）
- dotenv, langchain-openai, langgraph, langchain-coreなどの依存パッケージがインストールされていること

## 拡張性

- 新しいエージェントスキルを追加することで機能を拡張可能
- 異なる分析基準や閾値ロジックの実装が可能
- 市場価格取得専用のエージェントを追加することでより正確な分析が可能 