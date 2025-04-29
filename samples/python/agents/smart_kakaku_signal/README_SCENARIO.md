# シナリオベース異常検知システム

このシステムは自然言語で記述されたシナリオに基づいて柔軟に異常検知を行うことができます。
シナリオは固定のフォーマットではなく、自由な記述が可能で、LLMがその内容を解析して必要なデータと判定ロジックを特定します。

## 主な機能

1. **自然言語シナリオ**: 自由な文章で異常検知のシナリオを記述できます
2. **動的計画生成**: シナリオに基づいて必要なデータ収集の計画を自動生成します
3. **柔軟な判定ロジック**: 乖離率、価格差、数量比較など様々な条件に対応可能です
4. **シナリオ管理**: シナリオの保存、一覧表示、削除が可能です

## 使い方

### 起動オプション

- `--config <パス>` : 設定ファイルを指定して起動します（例: `--config agent_config.yaml`）
- `--update-config <key=value,...>` : 設定ファイルの一時的な上書きが可能です（例: `--update-config "agents.purchasing_data.url=http://localhost:5001"`）
- `--test-all` : すべてのテストシナリオを一括実行します
- `--log-file <パス>` : ログ出力先ファイルを指定します

> **PowerShellでの注意:**
> - JSONやイコール記号を含む引数はクォート（`'` または `"`）で囲んでください。
> - パス区切りは `\` ではなく `/` も利用可能です。

### シナリオの保存

```powershell
python -m agents.smart_kakaku_signal save \
  --id price_threshold \
  --name "価格閾値超過検知" \
  --description "指定された製品IDの販売価格が10000円を超える場合に異常と判定します。高額商品の販売状況を監視します。"
```

### シナリオ一覧の表示

```powershell
python -m agents.smart_kakaku_signal list
```

### シナリオの実行

#### 保存済みシナリオから実行

```powershell
python -m agents.smart_kakaku_signal run \
  --scenario-id price_deviation \
  --params '{"product_id": "P001", "threshold": 5.0}' \
  --log-file logs/smart_kakaku.log
```

#### 実行例
cd C:\Users\nyham\work\A2A\A2A\A2A_risk\samples\python; python -m agents.smart_kakaku_signal --log-file logs/smart_kakaku.log run --scenario-id scenario_test2


#### 直接シナリオを指定して実行

```powershell
python -m agents.smart_kakaku_signal run \
  --scenario-text "指定された製品IDの在庫数が5個未満の場合に異常と判定します。これは在庫不足の可能性があることを示します。" \
  --params '{"product_id": "P001"}' \
  --log-file logs/smart_kakaku.log
```

### シナリオの削除

```powershell
python -m agents.smart_kakaku_signal delete --id price_threshold
```

### 設定の一時的な上書き例

```powershell
python -m agents.smart_kakaku_signal --update-config "agents.purchasing_data.url=http://localhost:5001" run --scenario-id price_deviation --params '{"product_id": "P001"}'
```

### テストシナリオの一括実行

```powershell
python -m agents.smart_kakaku_signal --test-all --log-file logs/test_all.log
```

## シナリオの記述例

### 価格乖離検知

```
指定された製品IDの販売価格と市場価格の乖離率が設定されたしきい値（％）を超える場合、価格異常として検出します。これは販売価格が市場価格と大きく乖離している状況を示しています。
```

### 類似製品比較

```
指定された製品IDの販売価格が、類似製品（P002）の市場価格と1000円以内の差になっているかを判定します。差が1000円を超える場合は異常と判断します。
```

### 取引数量異常

```
指定された製品IDの取引数が他の製品の取引数の平均より半分以下であれば異常と判定します。これは商品の流通が少なく在庫が滞留している可能性があることを示します。
```

### カスタムシナリオ例

```
指定された製品の在庫回転率（月間販売数÷平均在庫数）が0.5未満であれば異常と判定します。これは商品の滞留を示し、販売戦略の見直しが必要かもしれません。
```

## パラメータの指定

シナリオ実行時に必要なパラメータをJSON形式で指定します。以下は主なパラメータ例です：

- `product_id`: 対象製品のID
- `threshold`: 異常判定の閾値（％）
- `reference_product_id`: 比較対象の製品ID
- `max_price_difference`: 許容される価格差（円）

## システムの拡張

新しいデータソースや判定ロジックに対応するには：

1. 新しいエージェントをA2Aエコシステムに追加
2. 適切なスキルを実装
3. `initialize_registry`関数でエージェントを登録

シナリオ実行エンジンがデータの取得と分析を自動的に行います。

## 従来方式との互換性

従来の方式（固定ロジックによる乖離率判定）も引き続き使用できます：

```bash
python -m samples.python.agents.smart_kakaku_signal legacy \
  --product_id P001 \
  --threshold 5.0
``` 