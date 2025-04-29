# 設定値・定数

# --- 汎用定数 ---
MAX_INCONCLUSIVE_BEFORE_GENERATE = 2
STAGNATION_THRESHOLD = 0.8
SIM_THRESHOLD = 3
EVAL_REPEAT_LIMIT = 10

# --- Verification 設定 ---
# verification フェーズの最大反復回数 (ループガード)
VERIFICATION_REPEAT_LIMIT = 10

# --- LLM設定 ---
DEFAULT_LLM_MODEL = "gpt-4.1-mini"
DEFAULT_LLM_TEMPERATURE = 0

# --- デフォルト設定 ---
DEFAULT_MAX_ITERATIONS = 50

# --- ロギング設定（必要に応じて main.py 側で利用） ---
LOG_FILE = 'test_react.log'
LOG_ENCODING = 'utf-8'
LOG_LEVEL = 'INFO' 