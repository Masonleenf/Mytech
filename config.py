# config.py
# 중앙화된 설정 관리

import os

# ============= MongoDB 설정 =============
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://rator9521_db_user:qwe343434@cluster0.d126rkt.mongodb.net/")

# 데이터베이스 이름
ETF_DATABASE = "etf_database"
ECOS_DATABASE = "ecos_database"

# 컬렉션 이름
COLLECTION_ETF_MASTER = "etf_master"
COLLECTION_ASSET_PAIRS = "asset_pairs"
COLLECTION_FUND_PRICES = "fund_prices"
COLLECTION_SYNTHETIC_INDICES = "synthetic_indices"
COLLECTION_MARKET_SUMMARY = "market_summary"
COLLECTION_ECOS_PRICES = "ecos_prices"

# 해외 배당 ETF 컬렉션 (dividen_model)
COLLECTION_DIVIDEND_ETF_SUMMARY = "dividend_etf_summary"
COLLECTION_DIVIDEND_ETF_PRICES = "dividend_etf_prices"
COLLECTION_DIVIDEND_PORTFOLIO = "dividend_portfolio"

# ============= 파일 경로 =============
DATA_DIR = "data"
LIST_CSV_PATH = os.path.join(DATA_DIR, "list.csv")
USER_CSV_FILE = os.path.join(DATA_DIR, "etf_info.csv")

# ============= 최적화 기본값 =============
DEFAULT_RISK_FREE_RATE = 0.02
DEFAULT_MIN_WEIGHT = 0.05
DEFAULT_MAX_WEIGHT = 0.65
DEFAULT_TRIM_RATIO = 0.3
