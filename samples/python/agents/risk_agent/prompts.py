import json
import datetime

def get_decision_maker_prompt(state, focused_hypothesis, currently_investigating_hypothesis_id, available_actions):
    objective = state['objective']
    # focused_hypothesis_for_promptのid,text,statusのみを取得する
    focused_hypothesis_for_prompt = {
        'id': focused_hypothesis['id'],
        'text': focused_hypothesis['text'],
        'status': focused_hypothesis['status'],
        'metric_definition': focused_hypothesis['metric_definition']
    }
    # 現在フォーカスしている仮説（currently_investigating_hypothesis_id）に紐づくobservationのみを抽出し、最新1件のみをプロンプト用に整形
    observation_history = [
        h for h in state.get('history', [])
        if h.get('type') == 'observation' and h.get('hypothesis_id') == currently_investigating_hypothesis_id
    ]
    history_for_prompt = json.dumps(observation_history[-1:], ensure_ascii=False, indent=2)
    available_agents = state.get('available_data_agents_and_skills', [])
    

    return f"""
### ROLE
あなたは経験豊富な内部監査人／調査エージェント AI です。仮説駆動型アプローチに従ってください。現在フォーカスしている仮説の検証を進めます。

### GOAL
{objective}

### CURRENT FOCUS (重要: 現在はこの仮説の検証に集中しています)
- CURRENTLY_INVESTIGATING_HYPOTHESIS: {focused_hypothesis_for_prompt}

### CONTEXT
- HISTORY_JSON: {history_for_prompt}

### AVAILABLE_RESOURCES
- ACTION_TYPES_JSON: {json.dumps(available_actions, ensure_ascii=False, indent=2)}
- DATA_AGENTS_JSON: {available_agents}

### ACTION-SELECTION POLICY (このポリシーに従ってください)
**Focus Hypothesis is {focused_hypothesis['status']}. Determine the next step:**
*   If focus_hypothesis.status == "new" → **must** select "evaluate_hypothesis" for this hypothesis_id ({currently_investigating_hypothesis_id}).
*   If focus_hypothesis.status ∈ {{"inconclusive"}}:
        *   Look at the latest entry in HISTORY_JSON.
        *   If the latest entry is an Observation from 'query_data_agent' related to the focus_hypothesis → **Analyze if the collected data (in observation.content.result) is sufficient to evaluate the hypothesis.**
            *   If **sufficient** → **must** select "evaluate_hypothesis" for this hypothesis_id ({currently_investigating_hypothesis_id}).
            *   If **insufficient** → **must** determine **what specific information is still missing** based on the focus_hypothesis.text, focus_hypothesis.required_next_data, and the collected data. Then, select "query_data_agent" with a **new, more specific query** to obtain the missing information. Avoid repeating the exact same query.
            *   To avoid errors from overly complex queries, **always start with a simple, first-step query that targets only the most essential missing information**.
            *   If multiple pieces of information are missing, **do not combine them into a single query**. Instead, address them one at a time, starting with the most fundamental or high-priority item.
        *   If the latest entry is NOT a relevant Observation → **must** select "query_data_agent" based on focus_hypothesis.required_next_data to get the initially requested data.
*   If focus_hypothesis.status == "investigating" → **must** select "evaluate_hypothesis" for this hypothesis_id ({currently_investigating_hypothesis_id}).
*   *Do not select "generate_hypothesis" or "conclude" while focusing on a hypothesis.*
*   **If the hypothesis is large or complex (complexity_score >= 0.7), you may select 'refine_plan' to break down the investigation into steps.**

### TASK
1. 上記 ACTION-SELECTION POLICY と状況に基づき、現在フォーカスしている仮説 ({currently_investigating_hypothesis_id}) の検証を進めるための *単一* の行動 (action_type) を選定し、必要パラメータを具体値で生成せよ。
2. 選定理由を "thought" フィールドに 1–3 文で要約せよ (POLICY をどう解釈したか、Focus の状態、履歴の考慮点を含めること)。
3. 失敗や不整合がある場合は action_type に "error" を指定せよ。
4. 各 action_type の parameters は以下の形式で出力せよ：
   - "evaluate_hypothesis": {{ "hypothesis_id": {currently_investigating_hypothesis_id} }}
   - "query_data_agent": {{ "agent_skill_id": "<データエージェントのID>", "query": "<具体的で新しい問い合わせ内容>" }}（queryには既取得情報を含めないこと）
   - "refine_plan": {{ "hypothesis_id": "{currently_investigating_hypothesis_id}" }}
   - "execute_plan_step": {{ "step_id": "<ステップID>" }}

### FEW-SHOT EXAMPLES
（以下の例は、すべて現在の仮説にフォーカスしている前提で記載されています）
Example 1: Status is inconclusive, no recent relevant observation.
{{
  "thought": "Currently focusing on '{currently_investigating_hypothesis_id}'. Its status is 'inconclusive' and it requires data 'XYZ' (from required_next_data field). The history does not show recent data collection for this. Policy requires querying the necessary data. Selecting 'query_data_agent'.",
  "action": {{
    "action_type": "query_data_agent",
    "parameters": {{ "agent_skill_id": "some_agent_skill", "query": "Retrieve XYZ data related to {currently_investigating_hypothesis_id} based on evaluation requirement." }}
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
    "parameters": {{ "agent_skill_id": "some_agent_skill", "query": "Retrieve specific field 'ABC' details for the records related to {currently_investigating_hypothesis_id} obtained in the previous query." }}
  }}
}}
Example 4: Status is complex, select refine_plan.
{{
  "thought": "Large/complex (complexity_score >= 0.7). Policy allows breaking down the investigation. Selecting 'refine_plan'.",
  "action": {{
    "action_type": "refine_plan",
    "parameters": {{ "hypothesis_id": "{currently_investigating_hypothesis_id}" }}
  }}
}}

### OUTPUT FORMAT (DO NOT WRAP IN CODE BLOCK)
{{
  "thought": "<why this action is optimal for {currently_investigating_hypothesis_id}, referencing the policy, focus state, and latest history observation>",
  "action": {{
    "action_type": "<one_of: {', '.join([a['type'] for a in available_actions if a['type'] not in ['generate_hypothesis', 'conclude']] )}, refine_plan, execute_plan_step>",
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
新規または改訂リスク仮説を生成し、優先度づけせよ。
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
4. しきい値や比較条件は必ず **数値＋単位** を含める (例: "1.5 倍", "20%", "50,000 円")。
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
    "time_window": {{ "start": "YYYY-MM-DD", "end": "YYYY-MM-DD" }},
    "supporting_evidence_keys": ["..."],
    "next_validation_step_suggestion": "...",
    "metric_definition": "<SQL/Pseudo-code for metric calculation>"
  }}, ...
]
    """

def get_evaluate_hypothesis_prompt(state, hypothesis_to_evaluate, parameters):
    observation_history = [
        h for h in state.get('history', [])
        if h.get('type') == 'observation' and h.get('hypothesis_id') == hypothesis_to_evaluate['id']
    ]
    history_for_prompt = json.dumps(observation_history, ensure_ascii=False, indent=2)
    # hypothesis_to_evaluate['id']と重複したキーのものだけに絞る
    collected_data_summary = state.get('collected_data_summary', {})
    filtered_data_summary = {k: v for k, v in collected_data_summary.items() if k == hypothesis_to_evaluate['id']}
    collected_data_summary_for_prompt = json.dumps(filtered_data_summary, ensure_ascii=False, indent=2)

    # 最新の data_analysis_result (該当仮説) を抽出
    latest_analysis_result = next((
        o['content']['data_analysis_result']
        for o in reversed(state.get('history', []))
        if o.get('type') == 'observation'
        and o.get('hypothesis_id') == hypothesis_to_evaluate['id']
        and isinstance(o.get('content'), dict)
        and 'data_analysis_result' in o['content']
    ), None)
    latest_analysis_result_json = json.dumps(latest_analysis_result, ensure_ascii=False, indent=2) if latest_analysis_result is not None else 'null'

    return f"""
### ROLE
あなたは客観的かつ批判的な監査評価者 AI です。
### TARGET_HYPOTHESIS
{json.dumps(hypothesis_to_evaluate, ensure_ascii=False, indent=2)}
### CONTEXT
- OBJECTIVE: {state['objective']}
- RELATED_HISTORY_JSON: {history_for_prompt}
- DATA_SUMMARY_JSON: {collected_data_summary_for_prompt}
- LATEST_DATA_ANALYSIS_RESULT_JSON: {latest_analysis_result_json}

### TASK
1. 利用可能なデータおよび最新のデータ分析結果に基づき、TARGET_HYPOTHESIS を評価せよ。
2. evaluation_status を supported | rejected | needs_revision | inconclusive から選べ。
3. reasoning は評価の根拠を 2–4 文で述べよ。
4. inconclusive または needs_revision の場合は、評価に必要な次のデータ (required_next_data) を具体的に示せ (null も可)。
   - データ分析結果に "取得できない" 旨の記述やレコード件数 0 / coverage=0 など十分なデータが得られていない兆候がある場合は、evaluation_status を "needs_revision" とし、hypothesis の見直しを推奨する理由を明記せよ。
### OUTPUT FORMAT (DO NOT WRAP IN CODE BLOCK)
{{ "hypothesis_id": "{hypothesis_to_evaluate['id']}", "evaluation_status": "<status>", "reasoning": "...", "required_next_data": "<null_or_specific_data_needed>" }}
    """

def get_refine_hypothesis_prompt(hypothesis):
    return f"""
あなたは監査人です。以下の仮説を、\n- 意味的に重複しない\n- 検証しやすいサブ仮説\nへ分割してください。\n{{ "hypothesis": {json.dumps(hypothesis, ensure_ascii=False)} }}
\n【必ずJSON形式（Pythonのlist of dict）で出力してください。コードブロックや説明文は不要です。】
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

SYSTEM:
You are an internal-audit hypothesis‐to‐rule selector AI.
以下の "PATTERN CATALOG" と "JSON SCHEMA" のルールに **厳密** に従い、  
以下の調査方針に対して、適切なパターン（複数可）とパラメータを **JSON 配列** で出力してください。  
調査方針：{query}

PATTERN CATALOG:
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
3. derive_month  
   • params:  
     - date_col (string)  
     - format (string, e.g. "%Y-%m")  
     - new_col (string)  
4. group_agg_generate  
   • params:  
     - source_col (string)  
     - group_cols (array of string)  
     - agg_func ("mean"|"sum"|"count")  
     - new_col (string)  
5. cross_compare  
   • params:  
     - left_ds (string)  
     - right_ds (string)  
     - join_keys (array of string)  
     - compare_cols (array of {{left:string, right:string}})  
     - tolerance_pct (number)  

JSON SCHEMA(answerキーにJSON配列を出力):
```json
"answer":
    [
    {{
    "step_id":"<1,2,3...>",
    "required_data":
        {{
            "new":<パターンに必要なデータ、カラムの一覧>,
            "other_step_output":<パターンに必要なデータ>
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

```

VALIDATION RULES:
- `pattern_id` は必ず上記 `enum` のいずれか
- `required_data` は'new'か'other_step_output'のいずれかを設定
- `other_step_output` は他のstepの出力結果を使用する場合は、「1」のように`step_id` のみを回答する
- `params` はそのパターンが定義する **必須キー** をすべて含む
- `params` に余分なキーを含めない
- 型チェックを必ず通過する（string/number/list）

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

def get_data_analysis_prompt(query, refined_query, data):
    return f"""

SYSTEM:
あなたは、データエージェントの実行結果を要約する AI アシスタントです。

### TASK
1. 以下のデータ分析の目的があります。
{query}

2. この目的を達成するためのデータ分析計画が以下です。
{refined_query}

3. 以下のデータを上記の方針計画を元にデータ分析して結果を回答してください。
{data}

### OUTPUT FORMAT(json形式で出力)
{{
  "data_analysis_result": "<データ分析の結果>",
  "reasoning": "<データ分析の根拠を述べる。>"
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