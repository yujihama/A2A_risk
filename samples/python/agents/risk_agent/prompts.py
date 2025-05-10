import json
import datetime

def get_decision_maker_prompt(state, focused_hypothesis, currently_investigating_hypothesis_id, available_actions):
    objective = state['objective']
    # focused_hypothesis_for_promptのid,text,statusのみを取得する
    focused_hypothesis_for_prompt = {
        'id': focused_hypothesis['id'],
        'text': focused_hypothesis['text'],
        'status': focused_hypothesis['status'],
        'metric_definition': focused_hypothesis['metric_definition'],
        'supporting_evidence_keys': focused_hypothesis['supporting_evidence_keys'],
        'next_validation_step_suggestion': focused_hypothesis['next_validation_step_suggestion'],
        'evaluation_reason': focused_hypothesis['evaluation_reason']
    }
    # 現在フォーカスしている仮説（currently_investigating_hypothesis_id）に紐づくobservationのみを抽出し、最新1件のみをプロンプト用に整形
    observation_history = [
        h for h in state.get('history', [])
        if h.get('currently_investigating_hypothesis_id') == currently_investigating_hypothesis_id
    ]
    history_for_prompt = json.dumps(observation_history[-3:], ensure_ascii=False, indent=2)
    
    collected_data_summary = state.get('collected_data_summary', {}).get(currently_investigating_hypothesis_id, {})
    collected_data_summary_for_prompt = json.dumps(collected_data_summary, ensure_ascii=False, indent=2)


    return f"""
### ROLE
あなたは経験豊富な内部監査人／調査エージェント AI です。仮説駆動型アプローチに従ってください。現在フォーカスしている仮説の検証を進めます。

### GOAL
{objective}

### CURRENT FOCUS (重要: 現在はこの仮説の検証に集中しています)
- CURRENTLY_INVESTIGATING_HYPOTHESIS: {focused_hypothesis_for_prompt}

### CONTEXT
- HISTORY_JSON: {history_for_prompt}
- COLLECTED_DATA_SUMMARY: {collected_data_summary_for_prompt}
### AVAILABLE_RESOURCES
- ACTION_TYPES_JSON: {json.dumps(available_actions, ensure_ascii=False, indent=2)}

### ACTION-SELECTION POLICY (このポリシーに従ってください)
**Focus Hypothesis is {focused_hypothesis['status']}. Determine the next step:**
*   If focus_hypothesis.status == "new" → **must** select "evaluate_hypothesis" for this hypothesis_id ({currently_investigating_hypothesis_id}).
*   If focus_hypothesis.status =="inconclusive":
        *   Look at the latest entry in HISTORY_JSON.
        *   If the latest entry is a hypothesis evaluation → **must** select "query_data_agent" and copy the content of focus_hypothesis.required_next_data directly into the query parameter to obtain the requested data.
*   If focus_hypothesis.status =="needs_revision":
        *   **If you need to revise the hypothesis, please select 'refine_hypothesis'.**
        *   **If the hypothesis is large or complex, you may select 'refine_hypothesis' to break down the investigation into steps.**
        
### TASK
1. 上記 ACTION-SELECTION POLICY と状況に基づき、現在フォーカスしている仮説 ({currently_investigating_hypothesis_id}) の検証を進めるための *単一* の行動 (action_type) を選定し、必要パラメータを具体値で生成せよ。
2. 選定理由を "thought" フィールドに 1–3 文で要約せよ (POLICY をどう解釈したか、Focus の状態、履歴の考慮点を含めること)。
3. 失敗や不整合がある場合は action_type に "error" を指定せよ。
4. 各 action_type の parameters は以下の形式で出力せよ：
   - "evaluate_hypothesis": {{ "hypothesis_id": {currently_investigating_hypothesis_id} }}
   - "query_data_agent": {{ "query": ["<具体的で新しい問い合わせ内容>",...] }}（queryには既取得情報を含めないこと）
   - "refine_hypothesis": {{ "hypothesis_id": "{currently_investigating_hypothesis_id}" }}

### FEW-SHOT EXAMPLES
（以下の例は、すべて現在の仮説にフォーカスしている前提で記載されています）
Example 1: Status is inconclusive, no recent relevant observation.
{{
  "thought": "Currently focusing on '{currently_investigating_hypothesis_id}'. Its status is 'inconclusive' and it requires data 'XYZ' (from required_next_data field). The history does not show recent data collection for this. Policy requires querying the necessary data. Selecting 'query_data_agent'.",
  "action": {{
    "action_type": "query_data_agent",
    "parameters": {{ "query": ["Retrieve XYZ data related to {currently_investigating_hypothesis_id} based on evaluation requirement."] }}
  }}
}}
Example 2: Status is inconclusive, **recent observation contains data 'XYZ'**.
{{
  "thought": "Status 'inconclusive'. The latest history entry is an observation containing relevant data 'XYZ' for this hypothesis. Policy requires re-evaluating the hypothesis with the new data. Selecting 'evaluate_hypothesis'.",
  "action": {{
    "action_type": "evaluate_hypothesis",
    "parameters": {{ "hypothesis_id": "{currently_investigating_hypothesis_id}" }}
  }}
}}
Example 3: Status inconclusive, **recent observation data is insufficient** (e.g., missing specific field 'ABC').
{{
  "thought": "Status 'inconclusive'. Recent observation provided some data, but analysis indicates field 'ABC' is still missing to fully evaluate. Policy requires querying for the missing specific information. Selecting 'query_data_agent' with a refined query.",
  "action": {{
    "action_type": "query_data_agent",
    "parameters": {{ "query": ["Retrieve specific field 'ABC' details for the records related to {currently_investigating_hypothesis_id} obtained in the previous query."] }}
  }}
}}
Example 4: Status is 'needs_revision', select refine_hypothesis.
{{
  "thought": "Status is 'needs_revision'. Policy allows revising the hypothesis. Selecting 'refine_hypothesis'.",
  "action": {{
    "action_type": "refine_hypothesis",
    "parameters": {{ "hypothesis_id": "{currently_investigating_hypothesis_id}" }}
  }}
}}

### OUTPUT FORMAT (DO NOT WRAP IN CODE BLOCK)
{{
  "thought": "<why this action is optimal for {currently_investigating_hypothesis_id}, referencing the policy, focus state, and latest history observation>",
  "action": {{
    "action_type": "<one_of: {', '.join(available_actions)}>",
    "parameters": {{<key_value_pairs_or_empty_object>}}
  }}
}}
    """

def get_generate_hypothesis_prompt(state):
    eda_summary = state.get('eda_summary', None)
    eda_stats = state.get('eda_stats', {})
    # 分析基準日を状態から取得、なければ今日の日付を使用
    analysis_date = state.get('analysis_date', datetime.date.today().isoformat())
    if eda_summary:
        eda_summary_lines = eda_summary.splitlines()
        eda_summary_head = "\n".join(eda_summary_lines[:5])
    else:
        eda_summary_head = None
    return f"""
### ROLE
あなたは洞察力のあるデータアナリスト／リスク分析官 AI です。
### GOAL
EDAの内容をもとに、以下のOBJECTIVEを検知するための仮説を2～5つ程度作成し、優先度づけせよ。
### CONTEXT
- ANALYSIS_DATE: {analysis_date}
- OBJECTIVE: {state['objective']}
- EDA_SUMMARY_TEXT: {json.dumps(eda_summary_head, ensure_ascii=False, indent=2) if eda_summary_head else 'null'}
- EDA_STATS_JSON: {json.dumps(eda_stats, ensure_ascii=False, indent=2)}
- RECENT_OBSERVATIONS_JSON: {json.dumps(state.get('history', [])[-3:], ensure_ascii=False, indent=2)}
- DATA_SUMMARY_JSON: {json.dumps(state.get('collected_data_summary', {}), ensure_ascii=False, indent=2)}
- EXISTING_HYPOTHESES_JSON: {json.dumps(state.get('current_hypotheses', []), ensure_ascii=False, indent=2)}
- AVAILABLE_DATA_AGENTS_JSON: {state.get('available_data_agents_and_skills', [])} 
### TASK
1. 0 個以上の仮説オブジェクトを生成せよ。既存仮説と **重複しない** ように考慮すること。
2. 各仮説は必ず IF … THEN … 形式の１文で記述し、疑問形は禁止。
3. 時間を示す語句は **直接的に数値で示せ**。例: "{analysis_date} から遡って 365 日間 (期間: 2024-04-28〜{analysis_date})" のように開始日・終了日を明示する。"最新" や "過去1年" など曖昧な語は禁止。
4. しきい値や比較条件は必ず **数値＋単位** を含める (例: "X 倍", "Y%", "Z 円")。
5. 各仮説には以下フィールドを含めよ: id (例: hyp_00X), text, priority (0–1), status='new', time_window={{"start":"YYYY-MM-DD","end":"YYYY-MM-DD"}}, supporting_evidence_keys, next_validation_step_suggestion, metric_definition.
6. priority は SMART 原則に基づき 0.5〜1.0 で設定。
7. EDA_SUMMARY_TEXT が null の場合は、まず代表サンプルを抽出して傾向把握する仮説を１件以上含めること。
8. 平均・最新などの指標を用いる場合は、その算出方法（集計単位・対象範囲・数式）を **metric_definition** フィールドで SQL 風あるいは Python 疑似コードで具体的に示すこと。例: `AVG(unit_price) OVER (PARTITION BY employee_id, vendor_id WHERE order_date BETWEEN '2024-04-28' AND '2025-04-28')`。
### OUTPUT FORMAT (hypothesisキーでJSON配列を出力)
"hypothesis": [
  {{
    "id": "hyp_###",
    "text": "IF ... THEN ...",
    "priority": 0.##,
    "status": "new",
    "time_window": "YYYY-MM-DD〜YYYY-MM-DD",
    "supporting_evidence_keys": ["..."],
    "next_validation_step_suggestion": "...",
    "metric_definition": "<SQL/Pseudo-code for metric calculation>"
  }}, ...
]
    """

def get_evaluate_hypothesis_prompt(state, hypothesis_to_evaluate, parameters):
    observation_history = [
        h for h in state.get('history', [])
        if ((h.get('type') == 'node' and h.get('content').get('name') in ('evaluate_hypothesis', 'query_data_agent')) or (h.get('type') in ('observation', 'thought'))) 
            and h.get('currently_investigating_hypothesis_id') == hypothesis_to_evaluate['id']            
    ]
    history_for_prompt = observation_history #json.dumps(observation_history, ensure_ascii=False, indent=2)
    
    # evaluate_hypothesisの結果がinconclusiveだった回数
    inconclusive_count = sum(1 for h in observation_history if h.get('type') == 'observation' and h.get('content').get('status') == 'inconclusive')
    
    collected_data_summary = state.get('collected_data_summary', {})
    filtered_data_summary = {k: v for k, v in collected_data_summary.items() if k == hypothesis_to_evaluate['id']}
    collected_data_summary_for_prompt = filtered_data_summary #json.dumps(filtered_data_summary, ensure_ascii=False, indent=2)

    # 最新の data_analysis_result (該当仮説) を抽出
    # 以前は history から逆順探索していたが、analysis_result に最新が保持されているため直接参照する。
    latest_analysis_result = None
    analysis_dict = state.get("analysis_result", {})
    if isinstance(analysis_dict, dict):
        latest_analysis_result = analysis_dict.get(hypothesis_to_evaluate['id'])

    latest_analysis_result_json = json.dumps(latest_analysis_result, ensure_ascii=False, indent=2) if latest_analysis_result is not None else 'null'
    
    hypothesis_to_evaluate_for_prompt = {
        "id": hypothesis_to_evaluate['id'],
        "text": hypothesis_to_evaluate['text'],
        "status": hypothesis_to_evaluate['status'],
        "time_window": hypothesis_to_evaluate['time_window'],
    }

    return f"""
### ROLE
あなたは客観的かつ批判的な監査評価者 AI です。
### TARGET_HYPOTHESIS
{hypothesis_to_evaluate_for_prompt}
### CONTEXT
- RELATED_HISTORY: {history_for_prompt}
- DATA_SUMMARY: {collected_data_summary_for_prompt}
- INCONCLUSIVE_COUNT: {inconclusive_count}

### TASK
1. 利用可能なデータおよび最新のデータ分析結果に基づき、TARGET_HYPOTHESIS を評価せよ。
2. evaluation_status を supported | rejected | needs_revision | inconclusive から選べ。
3. supportedと評価され場合、後続でさらに深堀りした仮説生成がされます。そのため、データから断言できなくても少しでも兆候が見られる場合はsupportedとしてください。
3. reasoning は評価の根拠を 2–4 文で述べよ。statusがsupportedの場合は、兆候が見られたデータを具体的を挙げて具体的に説明してください。
4. inconclusiveが今回含め3回目の場合、またはデータ取得が停滞していると判断される場合はneeds_revisionとすること。
4. inconclusive(十分なデータが取得できていない) の場合は、評価に必要な次のデータ (required_next_data) を具体的に示せ (null も可)。
   - required_next_data は、仮説検証に必要だが今回取得できていない情報を具体的に記載すること。
   
### OUTPUT FORMAT (JSON)
{{ "hypothesis_id": "{hypothesis_to_evaluate['id']}", "evaluation_status": "<status>", "reasoning": "...", "required_next_data": "<null_or_specific_data_needed>" }}

"""

# 以下コメントアウト
"""
### FEW-SHOT EXAMPLES
Example 1:
{{
  "hypothesis_id": "hyp_001",
  "evaluation_status": "supported",
  "reasoning": "...の理由により仮説は支持されると判断した。具体的には<データA>と<データB>...では...の傾向が見られ、...の兆候が見られた。",
  "required_next_data": null
}}
Example 2:
{{
  "hypothesis_id": "hyp_002",
  "evaluation_status": "rejected",
  "reasoning": "...の理由により仮説は支持されないと判断した。",
  "required_next_data": null
}}
Example 3:
{{
  "hypothesis_id": "hyp_003",
  "evaluation_status": "inconclusive",
  "reasoning": "...についてはデータが確認できたが、...についてはデータが確認できなかったため。",
  "required_next_data": "...と...についての....のデータ"
}}
Example 4:
{{
  "hypothesis_id": "hyp_004",
  "evaluation_status": "needs_revision",
  "reasoning": "データの取得を試みているが、エラーやタイムアウトのため取得できていないため。また、該当のデータないことが確認できたため。",
  "required_next_data": "SELECT 申請者ID, 申請金額, 申請日 FROM requests"
}}
    """

def get_refine_hypothesis_prompt(state, hypothesis):
    observation_history = [
        h for h in state.get('history', [])
        if h.get('type') == 'observation' and h.get('currently_investigating_hypothesis_id') == hypothesis['id']            
    ]
    hypothesis_for_prompt = {
        "id": hypothesis['id'],
        "text": hypothesis['text'],
        "time_window": hypothesis['time_window'],
        "supporting_evidence_keys": hypothesis['supporting_evidence_keys'],
        "metric_definition": hypothesis['metric_definition'],
    }

    return f"""
### ROLE
あなたは、監査の計画を作成するエージェントです。

### GOAL
以下の見直しが必要（needs_revision）と判断された仮説の見直しをして新しい仮説を生成してください。

### CONTEXT
- 見直す仮説: 
    {hypothesis_for_prompt}
- 見直しが必要となった理由: 
    {observation_history[-1].get('content').get('evaluation_reason')}

### TASK
1. 見直しが必要となった理由をよく理解し、仮説を見直してください。複数の仮説を生成することも可能です。
2. 各仮説は必ず IF … THEN … 形式の１文で記述し、疑問形は禁止。
3. 時間を示す語句は **直接的に数値で示せ**。例:開始日・終了日を明示する。"最新" や "過去1年" など曖昧な語は禁止。
4. しきい値や比較条件は必ず **数値＋単位** を含める (例: "X 倍", "Y%", "Z 円")。
5. 各仮説には以下フィールドを含めよ: id, text, priority (0–1), status='new', time_window={{"start":"YYYY-MM-DD","end":"YYYY-MM-DD"}}, supporting_evidence_keys, next_validation_step_suggestion, metric_definition.
6. priority は SMART 原則に基づき 0.5〜1.0 で設定。
7. 平均・最新などの指標を用いる場合は、その算出方法（集計単位・対象範囲・数式）を **metric_definition** フィールドで SQL 風あるいは Python 疑似コードで具体的に示すこと。例: `AVG(unit_price) OVER (PARTITION BY employee_id, vendor_id WHERE order_date BETWEEN '2024-04-28' AND '2025-04-28')`。

### OUTPUT FORMAT (hypothesisキーでJSON配列を出力)
"hypothesis": [
  {{
    "id": "hyp_###",
    "text": "IF ... THEN ...",
    "priority": 0.##,
    "status": "new",
    "time_window": "YYYY-MM-DD〜YYYY-MM-DD",
    "supporting_evidence_keys": ["..."],
    "next_validation_step_suggestion": "...",
    "metric_definition": "<SQL/Pseudo-code for metric calculation>"
  }}, ...
]

    """


def get_initial_data_query_prompt(objective, available_agents, analysis_date=None):
    """Generate prompt for selecting initial EDA data query.

    Parameters
    ----------
    objective : str
        Investigation objective.
    available_agents : list
        Available data agents.
    analysis_date : str | None
        Reference date (ISO8601). Defaults to today's date.
    """
    if analysis_date is None:
        analysis_date = datetime.date.today().isoformat()
    return f"""
### ROLE
あなたは、調査目的を達成するために最適な初期データ収集タスクを計画する AI アシスタントです。

### GOAL
提示された調査目的 (OBJECTIVE) のための初期データ分析 (EDA) に最も適したデータ収集タスク（単一）を決定してください。

### CONTEXT
- ANALYSIS_DATE: {analysis_date}
- OBJECTIVE: {objective}
- AVAILABLE_DATA_AGENTS_JSON: {available_agents} # 利用可能なデータソースとスキル

### TASK
1. 上記 OBJECTIVE と AVAILABLE_DATA_AGENTS_JSON を考慮し、初期 EDA に最も関連性が高く、基本的な情報を得られると考えられる *単一の* データエージェントスキル (`agent_skill_id`) を選択してください。
2. そのスキルに対して、OBJECTIVE に沿った初期データ取得のための具体的な問い合わせ内容 (`query`) を生成してください。
   - `query` には集計・分析に必要な最小限のカラムのみを含めること。
   - 時間条件を用いる場合は、必ず **期間の両端日 (YYYY-MM-DD)** を明示し、「最新」「過去〇年」など曖昧な語を使わないこと。
   - フィルタ条件や数値しきい値は **数値＋単位** で示すこと。
3. 生成する `query` は *シンプル* かつ *単一目的* とし、複数要件を詰め込まない。
4. 結果を以下の JSON 形式で出力してください。

### OUTPUT FORMAT (DO NOT WRAP IN CODE BLOCK)
{{
  "agent_skill_id": "<selected_agent_skill_id>",
  "query": "<generated_query_for_initial_data>"
}}
    """ 

def get_query_refinement_prompt(query):
    return f"""

### SYSTEM
You are an internal-audit hypothesis‐to‐rule selector AI.

### TASK
以下の "PATTERN CATALOG" と "JSON SCHEMA" のルールに **厳密** に従い、  
以下のデータを取得するために適切なパターン（複数可）とパラメータを **JSON 配列** で出力してください。  
必要なデータ：{query}

### PATTERN CATALOG
0. get_data
   • params:  
     - query (string ※SELECTから始まるSQLライクなクエリ)  

1. value_threshold  
   • params:  
     - target_col (string)  
     - mode ("static"|"ratio")  
     - static_value (number) ※mode=static の場合必須  
     - compare_col (string) ※mode=ratio の場合必須  
     - ratio_value (number) ※mode=ratio の場合必須  
     - operator (one of ">","=","<",">=","<=")  
2. duplicate_key  
   • params:  
     - key_cols (array of string)   
3. group_agg_generate  
   • params:  
     - source_col (string)  
     - group_cols (array of string)  
     - agg_func ("mean"|"sum"|"count")  
     - new_col (string)  
4. cross_compare  
   • params:  
     - left_ds (string)  
     - right_ds (string)  
     - join_keys (array of string)  
     - compare_cols (array of {{left:string, right:string}})  
     - tolerance_pct (number)  

### OUTPUT FORMAT (DO NOT WRAP IN CODE BLOCK)
  "answer":
    [
    {{
    "step_id":"<1,2,3...>",
    "required_data":
        {{
            "new":<パターンの実施に必要なインプットデータを新たにデータベースから取得する場合は、SELECTから始まるSQLライクなクエリ>,
            "other_step_output":<他のstepのアウトプットデータを使用する場合は、「1」のように`step_id` のみを回答する>
             }},
    "pattern_id":"<パターンID>",
    "params":{{<選択したパターンのパラメータ>}}
    }},
    {{
    "step_id":"<2,3,4...>",
    "required_data":{{<パターンに必要なデータ、カラムの一覧>}},
    "pattern_id":"<パターンID>",
    "params":{{<選択したパターンのパラメータ>}}
    }},
    ...
    ]

### VALIDATION RULES
- `pattern_id` は必ず上記 `enum` のいずれか
- `required_data` は'new'か'other_step_output'のいずれかを設定
- `other_step_output` は他のstepの出力結果を使用する場合は、「1」のように`step_id` のみを回答する
- `params` はそのパターンが定義する **必須キー** をすべて含む
- `params` に余分なキーを含めない
- 型チェックを必ず通過する（string/number/list）

### FEW-SHOT EXAMPLES

例1:
[
  {{
    "step_id": "1",
    "required_data": {{
      "new": "SELECT 申請者ID, 申請金額 FROM requests WHERE 申請金額 > 100000"
    }},
    "pattern_id": "group_agg_generate",
    "params": {{
      "source_col": "申請金額",
      "group_cols": ["申請者ID"],
      "agg_func": "count",
      "new_col": "高額申請件数"
    }}
  }},
  {{
    "step_id": "2",
    "required_data": {{
      "other_step_output": "1"
    }},
    "pattern_id": "group_agg_generate",
    "params": {{
      "source_col": "申請金額",
      "group_cols": ["申請者ID"],
      "agg_func": "sum",
      "new_col": "高額申請合計金額"
    }}
  }}
]

例2:
[
  {{
    "step_id": "1",
    "required_data": {{
      "new": "SELECT 申請者ID, 申請金額, 申請日 FROM requests"
    }},
    "pattern_id": "group_agg_generate",
    "params": {{
      "source_col": "申請金額",
      "group_cols": ["申請者ID"],
      "agg_func": "mean",
      "new_col": "平均申請金額"
    }}
  }},
  {{
    "step_id": "2",
    "required_data": {{
      "new": "SELECT 申請者ID, 申請金額, 申請日 FROM requests_昨年度"
    }},
    "pattern_id": "cross_compare",
    "params": {{
      "left_ds": "requests",
      "right_ds": "requests_昨年度",
      "join_keys": ["申請者ID"],
      "compare_cols": [
        {{"left": "申請金額", "right": "申請金額"}}
      ],
      "tolerance_pct": 5
    }}
  }}
]


    """

def get_query_result_summary_prompt(result_text, required_data_list):
    return f"""

SYSTEM:
あなたは、データエージェントの実行結果を要約する AI アシスタントです。

### TASK
1. 以下のrequired_data_listを取得するためのデータエージェントの実行結果を評価してください。
required_data_list:
{json.dumps(required_data_list, ensure_ascii=False, indent=2)}

2. 実行結果は以下の通りです。
{result_text}


### OUTPUT FORMAT(json形式で出力)
{{
  "summary": "<実行結果の要約>",
  "judgement": "<required_data_listの情報を漏れなく取得できているかの判定。OK/NG>",
  "reasoning": "<judgementの理由を述べる。>"
}}
    """

def get_eda_prompt(query, data):
    return f"""

SYSTEM:
あなたは、データエージェントの実行結果を要約する AI アシスタントです。

### TASK
1. 以下のEDA計画があります。
{query}

2. 以下のデータを上記の方針を元にEDAしてください。
{data}

### OUTPUT FORMAT(json形式で出力)
{{
  "eda_result": "<EDAの結果>",
}}
    """

def get_query_result_prompt(query, refined_query, data):
    return f"""

SYSTEM:
あなたは、データエージェントの実行結果を要約する AI アシスタントです。

### TASK
1. 以下のデータ分析の目的があります。
{query}

2. この目的を達成するためのデータ分析計画が以下です。
{refined_query}

3. 以下のデータを使って、上記の分析計画を元に結果を生成してください。
{data}

### OUTPUT FORMAT(json形式で出力)
{{
  "data_analysis_result": "<データ分析結果>",
  "summary": "<データ分析結果の要約>"
}}
    """

def get_query_data_analysis_prompt(query, output_from_data_agent):

    return f"""

SYSTEM:
あなたは、データサイエンティストです。

### TASK
1. 以下のデータ分析して結果を生成するという目的があります。
{query}

2. この目的を達成するために取得したデータは以下の通りです。
{output_from_data_agent}

3. データ分析の結果を生成してください。与えられたデータから結果を生成できない場合はその旨を回答してください。

### OUTPUT FORMAT(json形式で出力)
{{
  "data_analysis_result": "<データ分析の結果>",
  "reasoning": "<データ分析の根拠を述べる。結果を生成できない場合はその旨を述べる。>"
}}
    """

# ---------------- Verification Phase Prompts ----------------

def get_plan_verification_prompt(hypothesis, objective, available_agents):
    """Return a prompt template asking LLM to propose verification steps.

    Expected structured JSON output:
    {
      "thought": "short explanation",
      "verification_plan": [
          {"step_id": "v1", "agent_skill_id": "skill_x", "query": "..."},
          ...
      ]
    }
    """
    return f"""
### ROLE
あなたはフォレンジック・データアナリスト AI です。次の supported 仮説を追加データで裏付けるためのプランを立案します。

### GOAL (調査目的)
{objective}

### HYPOTHESIS TO VERIFY
{json.dumps(hypothesis, ensure_ascii=False, indent=2)}

### AVAILABLE DATA AGENTS & SKILLS
{json.dumps(available_agents, ensure_ascii=False, indent=2)}

### TASK
1. 仮説を補強・反証するのに最も効果的なデータポイントを 3 件以内で特定せよ。
2. 各データポイントについて、どのエージェントのどのスキルを利用し、どのようなクエリを送るかを具体的に設計せよ。
3. 出力は verification_plan 配列に step_id, agent_skill_id, query を含めること。step_id は "v1", "v2" ... のように付番。

### OUTPUT FORMAT (DO NOT WRAP IN CODE BLOCK)
{{
  "thought": "<簡潔な方針説明>",
  "verification_plan": [
    {{"step_id": "v1", "agent_skill_id": "<skill>", "query": "<query string>"}},
    ...
  ]
}}
    """

# --- summarize_verification_prompt ---

def get_summarize_verification_prompt(hypothesis, collected_results):
    return f"""
### ROLE
あなたは調査結果を総合評価するフォレンジック監査 AI です。

### HYPOTHESIS
{json.dumps(hypothesis, ensure_ascii=False, indent=2)}

### NEWLY COLLECTED VERIFICATION DATA
{json.dumps(collected_results, ensure_ascii=False, indent=2)}

### TASK
1. 上記データが仮説をどの程度裏付けているかを評価し、risk_score (0:安全〜1:極めて高いリスク) を与えよ。
2. verification_status を verified | disproved | pending のいずれかで選択。
3. reasoning を 2–4 文で要約。

### OUTPUT FORMAT (DO NOT WRAP IN CODE BLOCK)
{{"verification_status":"<status>","risk_score":0.##,"reasoning":"..."}}
    """

def get_query_data_first_step_prompt(state,query_list):
    available_agents = state.get("available_data_agents_and_skills", [])
    available_agents_json = json.dumps(available_agents, ensure_ascii=False, indent=2)

    return f"""
### ROLE
あなたはフォレンジック・データアナリスト AI です。次の TASK を達成するためのデータ取得計画を立案します。

### TASK
1. 以下のデータをステップバイステップで取得するという目的があります。
{query_list}

2. あなたが使えるデータエージェントは以下です。
{available_agents_json}

3. 目的を達成するために、まずはじめにどのデータエージェントからデータを取得するかを選択してください。

### OUTPUT FORMAT (Answer in JSON format)
{{
  "skill_id": "<selected_agent_skill_id>",
  "query": "<generated_query_for_initial_data>",
  "reasoning": "<選択理由を述べる。>"
}}
    """

def get_query_data_step_prompt(state,query_list,data_plan_list):
    available_agents = state.get("available_data_agents_and_skills", [])
    available_agents_json = json.dumps(available_agents, ensure_ascii=False, indent=2)

    data_plan_list_json = json.dumps(data_plan_list, ensure_ascii=False, indent=2)

    return f"""

SYSTEM:
あなたはフォレンジック・データアナリスト AI です。次の TASK を達成するためのデータ取得計画を立案します。

### TASK
1. 以下のデータをステップバイステップで取得するという目的があります。
{query_list} 

2. 以下のCONTEXTをよく読み、目的を達成するために次のステップとしてどのデータエージェントからどのデータを取得するかを選択してください。
既に目的の達成に必要なデータを取得していると思われる場合はskill_idとqueryを空にしてください。

### CONTEXT
- 利用可能なデータエージェント
{available_agents_json}

- これまでのデータ取得状況
{data_plan_list_json}

### OUTPUT FORMAT (Answer in JSON format)
{{
  "skill_id": "<selected_agent_skill_id or empty(if already collected)>",
  "query": "<generated_query_for_data_collection or empty(if already collected)>",
  "reasoning": "<選択理由を述べる。>"
}}

### Notice
- queryは独立したクエリであること。これまで取得したデータや状況を参照するようなクエリは作成不可。

    """

# --- Supporting Hypothesis Generation Prompt ---

def get_supporting_hypothesis_prompt(state, parent_hypothesis):
    """
    Supported となった親仮説に基づいて、それを深掘りまたは裏付けるための
    追加仮説を生成するためのプロンプトを生成します。

    Args:
        state (Dict): 現在のエージェントの状態。
        parent_hypothesis (Dict): 支持された親仮説。

    Returns:
        str: LLM向けのプロンプト文字列。
    """
    parent_hypothesis_json = json.dumps({
        "id": parent_hypothesis.get("id"),
        "text": parent_hypothesis.get("text")
    }, ensure_ascii=False, indent=2)

    parent_hypothesis_reasoning_json = json.dumps(parent_hypothesis.get("evaluation_reason"), ensure_ascii=False, indent=2)

    # 親仮説の評価に使われた分析結果を取得
    analysis_result_json = json.dumps(state.get('analysis_result', {}).get(parent_hypothesis.get("id"), {}), ensure_ascii=False, indent=2)
    # 利用可能なデータソース情報を取得
    available_agents_json = json.dumps(state.get('available_data_agents_and_skills', []), ensure_ascii=False, indent=2)

    return f"""
### ROLE
あなたは洞察力のある調査分析官 AI です。提示された仮説が支持されたことを受け、その確証度を高めるための追加仮説を考案します。

### GOAL
以下のPARENT_HYPOTHESISをさらに裏付ける、あるいはその背景要因を深掘りするための、具体的で検証可能な追加仮説を生成してください。
- PARENT_HYPOTHESIS (Supported): {parent_hypothesis_json} # この仮説を深掘りします
- PARENT_HYPOTHESIS_EVALUATION_REASON: {parent_hypothesis_reasoning_json} # 親仮説が支持された根拠: 

### CONTEXT
- OBJECTIVE: {state['objective']}
- AVAILABLE_DATA_AGENTS_AND_SKILLS # 利用可能なデータソースとスキル（追加仮説の着想源）:
{available_agents_json} 

### TASK
1. PARENT_HYPOTHESISの内容、およびそれが支持された根拠 (PARENT_HYPOTHESIS_EVALUATION_REASON) を踏まえてください。
2. その上で、**利用可能な他のデータソース (AVAILABLE_DATA_AGENTS_AND_SKILLS を参照) も考慮**し、親仮説で支持された**具体的な根拠を補強する**具体的な追加仮説を 1〜3 件生成してください。
3. **着眼点のヒント:** 親仮説に関連する人物やエンティティの**他の行動**、関連する**プロセスの不備**（例：承認プロセス）、**比較対象との差異**（例：同僚、他部署、過去期間）、データの**時間的な変化**や**相関関係**などに着目すると、有効な追加仮説が見つかることがあります。
4. 各仮説は、PARENT_HYPOTHESIS とは異なる**新しい視点**を提供する必要があります。
5. 各仮説は、**IF (条件) THEN (結果)** 形式の明確なステートメントで記述してください（疑問形は不可）。
6. 各仮説には以下を含めてください:
    - `id`: 新しいユニークなID (例: sub_hyp_001)
    - `text`: IF...THEN... 形式の仮説文
    - `priority`: 0.0〜1.0 の優先度 (親仮説の重要度や、裏付けの必要性から判断)
    - `status`: "new" (固定)
    - `parent_hypothesis_id`: "{parent_hypothesis.get('id')}" (固定 - この値は変更しないでください)
    - `time_window` (Optional): 親仮説の時間枠を引き継ぐか、必要なら再定義 ({{"start":"YYYY-MM-DD","end":"YYYY-MM-DD"}})
    - `supporting_evidence_keys` (Optional): 検証に必要となりそうなデータキーのリスト (AVAILABLE_DATA_AGENTS_AND_SKILLS を参考に具体的に)
    - `next_validation_step_suggestion` (Optional): 検証の次のステップの提案
    - `metric_definition` (Optional): 仮説内の指標の定義 (SQL/Pseudo-code)
7. 時間範囲やしきい値を含む場合は、具体的かつ明確な数値で示してください。曖昧な表現は避けてください。
"""

# 一時的にコメントアウト
"""
### FEW-SHOT EXAMPLE (着想の参考にしてください)
(親仮説: 「従業員Aの経費精算で交際費が急増している」が Supported)
{{ "hypotheses": [
    {{
      "id": "sub_hyp_001",
      "text": "IF 従業員Aの高額交際費申請が、対応する承認者の承認記録において申請から5分以内に行われている割合が50%以上 THEN 承認プロセスが形骸化している可能性がある",
      "priority": 0.8,
      "status": "new",
      "parent_hypothesis_id": "parent_hyp_123",
      "supporting_evidence_keys": ["経費精算データ", "ワークフロー承認ログ"],
      "metric_definition": "COUNT(CASE WHEN approval_timestamp - request_timestamp <= interval '5 minutes' THEN 1 END) / COUNT(*) > 0.5 PARTITION BY employee_id='A', expense_type='交際費'"
    }},
    {{
      "id": "sub_hyp_002",
      "text": "IF 従業員Aの高額交際費申請の日時に、従業員Aのカレンダーに関連する会議やアポイントメント記録が存在しない割合が30%以上 THEN 申請内容の信憑性に疑いがある",
      "priority": 0.7,
      "status": "new",
      "parent_hypothesis_id": "parent_hyp_123",
      "supporting_evidence_keys": ["経費精算データ", "カレンダーデータ"]
    }}
]}}

### OUTPUT FORMAT (必ず "hypotheses" キーを持つ JSON リスト形式で出力してください)
{{
  "hypotheses": [
    {{
      "id": "sub_hyp_###",
      "text": "IF [追加の条件/観測 from 関連データ] THEN [PARENT_HYPOTHESIS を補強する、または新たな側面を示す結果]",
      "priority": 0.##,
      "status": "new",
      "parent_hypothesis_id": "{parent_hypothesis.get('id')}",
      "time_window": "YYYY-MM-DD〜YYYY-MM-DD", // Optional
      "supporting_evidence_keys": ["..."], // Optional & Specific
      "next_validation_step_suggestion": "...", // Optional
      "metric_definition": "<SQL/Pseudo-code>" // Optional
    }},
    // ... (さらに追加仮説があれば)
  ]
}}
    """
def get_query_data_react_prompt(state,query_list):
    available_agents = state.get("available_data_agents_and_skills", [])
    available_agents_json = json.dumps(available_agents, ensure_ascii=False, indent=2)

    return f"""
### ROLE
あなたはフォレンジック・データアナリスト AI です。次の TASK を達成するためのデータ取得計画を立案します。

### TASK
1. 以下のデータをステップバイステップで取得するという目的があります。
{query_list}

2. あなたが使えるデータエージェントは以下です。
{available_agents_json}

3. 目的をデータを漏れなく取得するために、データエージェントを使って漏れなく正確にデータを取得してください。
どのデータエージェントを使っても目的のデータが取得できない場合はその旨を回答してください。

### OUTPUT FORMAT (Answer in JSON format)
{{
  "data":<取得したデータ>,
  "description":<取得したデータの説明>,
  "background":<取得経緯>
}}
    """


