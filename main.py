from flask import Flask, jsonify, request
from flask_cors import CORS
import optimizer as optimizer
import backtester as backtester
import json
import os
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
import beg_optimize
import market_ranking

app = Flask(__name__)
CORS(app)

# ============= MongoDB 설정 =============
MONGO_URI = "mongodb+srv://rator9521_db_user:qwe343434@cluster0.d126rkt.mongodb.net/"
ETF_DATABASE = "etf_database"
ECOS_DATABASE = "ecos_database"

client = MongoClient(MONGO_URI)
etf_db = client[ETF_DATABASE]
ecos_db = client[ECOS_DATABASE]

etf_master_collection = etf_db['etf_master']
asset_pairs_collection = etf_db['asset_pairs']
fund_prices_collection = etf_db['fund_prices']
synthetic_indices_collection = etf_db['synthetic_indices']
ecos_prices_collection = ecos_db['ecos_prices']

DATA_DIR = "data"
LIST_CSV_PATH = os.path.join(DATA_DIR, "list.csv")

def load_list_csv():
    """list.csv 파일을 로드하여 item_code1과 통계 정보 매핑"""
    try:
        if not os.path.exists(LIST_CSV_PATH):
            print(f"⚠ list.csv 파일이 없습니다: {LIST_CSV_PATH}")
            return {}
        
        df = pd.read_csv(LIST_CSV_PATH)
        df.columns = df.columns.str.strip()
        
        mapping = {}
        for _, row in df.iterrows():
            item_code1 = int(row['item_code1'])
            mapping[item_code1] = {
                'stat_code': str(row['stat_code']).strip(),
                'name': str(row['name']).strip(),
                'period': str(row['period']).strip(),
                'unit': str(row['단위']).strip()
            }
        
        return mapping
        
    except Exception as e:
        print(f"⚠ list.csv 로드 실패: {e}")
        return {}

def get_ecos_data_from_mongodb(item_code1):
    """MongoDB에서 특정 item_code1의 최신 2일 데이터 조회"""
    try:
        doc = ecos_prices_collection.find_one({'item_code1': item_code1})
        
        if not doc or 'prices' not in doc or not doc['prices']:
            return None
        
        df = pd.DataFrame(doc['prices'])
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d', errors='coerce')
        df = df.dropna(subset=['Date', 'Close'])
        df = df.sort_values('Date')
        
        if len(df) < 1:
            return None
        
        latest_data = df.tail(2)
        
        latest = {
            'TIME': latest_data.iloc[-1]['Date'].strftime('%Y%m%d'),
            'DATA_VALUE': str(latest_data.iloc[-1]['Close'])
        }
        
        previous = None
        if len(latest_data) >= 2:
            previous = {
                'TIME': latest_data.iloc[-2]['Date'].strftime('%Y%m%d'),
                'DATA_VALUE': str(latest_data.iloc[-2]['Close'])
            }
        
        return latest, previous
        
    except Exception as e:
        return None

def get_latest_market_data(stat_code, item_code1, list_mapping):
    """특정 통계의 최신 2일 데이터 조회"""
    try:
        if item_code1 not in list_mapping:
            return None
        
        stat_info = list_mapping[item_code1]
        
        mongodb_result = get_ecos_data_from_mongodb(item_code1)
        if not mongodb_result:
            return None
        
        latest, previous = mongodb_result
        
        current_value = float(latest['DATA_VALUE'])
        previous_value = float(previous['DATA_VALUE']) if previous else current_value
        
        change = current_value - previous_value
        change_percent = (change / previous_value * 100) if previous_value != 0 else 0
        
        if change > 0:
            trend = 'up'
        elif change < 0:
            trend = 'down'
        else:
            trend = 'neutral'
        
        return {
            'name': stat_info['name'],
            'value': str(current_value),
            'change': f"{change:+.2f}",
            'changePercent': f"{change_percent:+.2f}%",
            'trend': trend,
            'unit': stat_info.get('unit', ''),
            'lastUpdated': latest['TIME'],
            'stat_code': stat_code,
            'item_code1': item_code1
        }
        
    except Exception as e:
        return None

def format_market_indicators():
    """시장지표 데이터를 프론트엔드 형식으로 변환"""
    list_mapping = load_list_csv()
    
    if not list_mapping:
        return {
            'interest_rates': [],
            'stock_indices': [],
            'exchange_rates': []
        }
    
    market_indicators = {
        'interest_rates': [
            ('817Y002', 10101000, 'percent', '콜금리', '한국은행 기준금리'),
            ('817Y002', 10150000, 'percent', 'KORIBOR 3M', '3개월 금리'),
            ('817Y002', 10190000, 'percent', '국고채(1년)', '1년 국채 수익률'),
            ('817Y002', 10195000, 'percent', '국고채(2년)', '2년 국채 수익률'),
            ('817Y002', 10200000, 'percent', '국고채(3년)', '3년 국채 수익률'),
            ('817Y002', 10200001, 'percent', '국고채(5년)', '5년 국채 수익률'),
            ('817Y002', 10210000, 'percent', '국고채(10년)', '10년 국채 수익률'),
            ('817Y002', 10220000, 'percent', '국고채(20년)', '20년 국채 수익률'),
            ('817Y002', 10230000, 'percent', '국고채(30년)', '30년 국채 수익률'),
            ('817Y002', 10240000, 'percent', '국고채(50년)', '50년 국채 수익률'),
            ('817Y002', 10300000, 'percent', '회사채(3년, AA-)', 'AA- 회사채'),
            ('817Y002', 10320000, 'percent', '회사채(3년, BBB-)', 'BBB- 회사채'),
        ],
        'stock_indices': [
            ('802Y001', 1000, 'trending-up', 'KOSPI', '코스피 지수'),
            ('802Y001', 89000, 'trending-up', 'KOSDAQ', '코스닥 지수'),
        ],
        'exchange_rates': [
            ('731Y001', 1, 'dollar-sign', 'USD/KRW', '미국달러'),
            ('731Y001', 53, 'dollar-sign', 'CNY/KRW', '위안'),
            ('731Y001', 2, 'dollar-sign', 'JPY/KRW', '엔화(100)'),
            ('731Y001', 3, 'dollar-sign', 'EUR/KRW', '유로'),
            ('731Y001', 12, 'dollar-sign', 'GBP/KRW', '파운드'),
            ('731Y001', 13, 'dollar-sign', 'CAD/KRW', '캐나다달러'),
            ('731Y001', 17, 'dollar-sign', 'AUD/KRW', '호주달러'),
        ]
    }
    
    result = {}
    
    for category, indicators in market_indicators.items():
        category_data = []
        
        for stat_code, item_code1, icon_type, display_name, description in indicators:
            data = get_latest_market_data(stat_code, item_code1, list_mapping)
            
            if data:
                if category == 'interest_rates':
                    value_display = f"{float(data['value']):.2f}%"
                    change_bp = float(data['change']) * 100
                    change_display = f"{change_bp:+.0f}bp"
                elif category == 'stock_indices':
                    value_display = f"{float(data['value']):,.1f}"
                    change_display = f"{float(data['change']):+.1f}"
                else:
                    value_display = f"{float(data['value']):,.2f}원"
                    change_display = f"{float(data['change']):+.1f}"
                
                formatted_data = {
                    'id': f"{stat_code}_{item_code1}",
                    'icon': icon_type,
                    'name': display_name,
                    'description': description,
                    'value': value_display,
                    'change': change_display,
                    'changePercent': data['changePercent'],
                    'trend': data['trend'],
                    'unit': data.get('unit', ''),
                    'lastUpdated': data['lastUpdated']
                }
                
                category_data.append(formatted_data)
        
        result[category] = category_data
    
    return result

def get_etf_price_info(ticker):
    """MongoDB에서 ETF 가격 정보 조회"""
    try:
        doc = fund_prices_collection.find_one({'ticker': ticker})
        
        if not doc or 'prices' not in doc or not doc['prices']:
            return None
        
        df = pd.DataFrame(doc['prices'])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna(subset=['Close'])
        
        if len(df) < 2:
            return None
        
        latest_data = df.tail(2)
        
        current_price = latest_data['Close'].iloc[-1]
        previous_price = latest_data['Close'].iloc[-2]
        
        price_change = current_price - previous_price
        price_change_rate = (price_change / previous_price) * 100 if previous_price != 0 else 0
        
        return {
            "current_price": round(current_price, 2),
            "previous_price": round(previous_price, 2),
            "price_change": round(price_change, 2),
            "price_change_rate": round(price_change_rate, 2),
            "last_updated": latest_data.index[-1].strftime('%Y-%m-%d')
        }
        
    except Exception as e:
        return None

def perform_portfolio_optimization(asset_pairs, params):
    """공통 포트폴리오 최적화 로직 - Beginner, MVO, RiskParity, Rebalancing 모드 지원"""
    try:
        # MongoDB에서 ETF 마스터 로드
        from pymongo import MongoClient
        
        MONGO_URI = "mongodb+srv://rator9521_db_user:qwe343434@cluster0.d126rkt.mongodb.net/"
        client = MongoClient(MONGO_URI)
        db = client["etf_database"]
        etf_master_collection = db['etf_master']
        
        # 모드 확인
        mode = params.get("mode", "MVO")
        
        print(f"============================================================")
        print(f"📥 받은 요청 데이터:")
        print(f"  - mode: {mode}")
        print(f"  - asset_pairs: {asset_pairs}")
        print(f"  - optimization_params: {params}")
        print(f"============================================================")
        
        # ==================== Beginner 모드 ====================
        if mode == "Beginner":
            print("\n" + "="*60)
            print(" Beginner 모드 - beg_optimize 호출 ".center(60, "="))
            print("="*60)
            
            style_index = params.get("style_index")
            risk_index = params.get("risk_index")
            
            if style_index is None or risk_index is None:
                raise ValueError("Beginner 모드에는 style_index와 risk_index가 필요합니다.")
            
            print(f"  - style_index: {style_index}")
            print(f"  - risk_index: {risk_index}")
            
            # beg_optimize.py의 get_beginner_portfolio() 호출
            # 모든 비즈니스 로직은 beg_optimize.py에서 처리
            selected_tickers, weights, performance = beg_optimize.get_beginner_portfolio(
                style_index=style_index,
                risk_index=risk_index
            )
            
            print(f"✅ beg_optimize 결과 수신: {len(selected_tickers)}개 ETF")
            
            return selected_tickers, weights, performance
        
        # ==================== 이하 기존 모드 (MVO, RiskParity, Rebalancing) ====================
        
        # MongoDB에서 데이터 로드
        etf_data = list(etf_master_collection.find({}, {'_id': 0}))
        if not etf_data:
            raise FileNotFoundError("ETF 마스터 데이터가 없습니다.")
        
        etf_df = pd.DataFrame(etf_data)
        
        # 리밸런싱 모드 확인
        current_weights = params.get("current_weights", {})
        
        if mode == "Rebalancing" and current_weights:
            print(f"✅ current_weights를 params에 추가: {current_weights}")
            
            # 리밸런싱 모드: ticker와 code 분리
            holding_tickers = list(current_weights.keys())
            
            print(f"🔍 리밸런싱 모드 - 혼합 방식")
            print(f"  📊 보유 종목 (ticker): {holding_tickers}")
            
            # 추가 자산 처리 (asset_pairs에서 새로운 자산만)
            selected_codes = []
            added_assets = []
            
            for pair in asset_pairs:
                saa = pair.get("saa_class")
                taa = pair.get("taa_class")
                
                if saa == "EXISTING":
                    continue  # 기존 보유 종목은 이미 holding_tickers에 있음
                    
                matched_etf = etf_df[
                    (etf_df['saa_class'] == saa) & 
                    (etf_df['taa_class'] == taa)
                ]
                
                if not matched_etf.empty:
                    code = matched_etf['code'].iloc[0]
                    ticker = matched_etf['ticker'].iloc[0]
                    selected_codes.append(code)
                    added_assets.append(f"{saa} - {taa}")
                    print(f"  ➕ 추가 자산: [{saa} - {taa}] → code: {code}, ticker: {ticker}")
            
            print(f"  📈 추가 자산 (code): {selected_codes}")
            
            # code_to_ticker_map 생성
            code_to_ticker_map = {}
            for code in selected_codes:
                matched_etf = etf_df[etf_df['code'] == code]
                if not matched_etf.empty:
                    ticker = matched_etf['ticker'].iloc[0]
                    code_to_ticker_map[code] = ticker
            
            print(f"==================================================")
            
            # 리밸런싱 전용 함수 호출
            weights, performance = optimizer.get_optimized_portfolio_rebalancing(
                holding_tickers, selected_codes, code_to_ticker_map, params
            )
            
            # selected_codes는 결과 표시용 (ticker 형태로 변환)
            result_codes = holding_tickers.copy()
            for code in selected_codes:
                ticker = code_to_ticker_map.get(code, f"{code}.KS")
                if ticker not in result_codes:
                    result_codes.append(ticker)
            
            return result_codes, weights, performance
            
        else:
            # 일반 최적화 모드 (MVO, RiskParity)
            print(f"🔍 일반 최적화 모드: {mode}")
            selected_codes = []
            
            for pair in asset_pairs:
                saa = pair.get("saa_class")
                taa = pair.get("taa_class")
                
                matched_etf = etf_df[
                    (etf_df['saa_class'] == saa) & 
                    (etf_df['taa_class'] == taa)
                ]
                
                if not matched_etf.empty:
                    code = matched_etf['code'].iloc[0]
                    selected_codes.append(code)
                    print(f"  조합 ['{saa}' - '{taa}'] 대표 코드: {code}")
                else:
                    print(f"  경고: 조합 ['{saa}' - '{taa}']에 해당하는 ETF가 없습니다.")
            
            if len(selected_codes) < 2:
                raise ValueError("최적화를 위해 2개 이상의 종목이 필요합니다.")
            
            # code_to_ticker_map 생성
            code_to_ticker_map = {}
            for code in selected_codes:
                matched_etf = etf_df[etf_df['code'] == code]
                if not matched_etf.empty:
                    ticker = matched_etf['ticker'].iloc[0]
                    code_to_ticker_map[code] = ticker
            
            print(f"  📈 선택된 코드: {selected_codes}")
            
            # 일반 최적화 함수 호출
            weights, performance = optimizer.get_optimized_portfolio(
                selected_codes, params, code_to_ticker_map
            )
            
            # 결과 표시용으로 ticker로 변환
            result_codes = []
            for code in selected_codes:
                ticker = code_to_ticker_map.get(code, f"{code}.KS")
                result_codes.append(ticker)
            
            return result_codes, weights, performance
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "API 서버가 정상 작동 중입니다."}), 200

@app.route('/api/market-indicators', methods=['GET'])
def get_market_indicators():
    try:
        indicators = format_market_indicators()
        return jsonify({
            "status": "success",
            "data": indicators,
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"시장지표 조회 중 오류 발생: {str(e)}"
        }), 500

@app.route('/api/market-indicators/<category>', methods=['GET'])
def get_market_indicators_by_category(category):
    try:
        valid_categories = ['interest_rates', 'stock_indices', 'exchange_rates']
        
        if category not in valid_categories:
            return jsonify({
                "status": "error", 
                "message": f"유효하지 않은 카테고리입니다."
            }), 400
        
        indicators = format_market_indicators()
        
        return jsonify({
            "status": "success",
            "category": category,
            "data": indicators.get(category, []),
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"시장지표 조회 중 오류 발생: {str(e)}"
        }), 500

@app.route('/api/market-indicators/summary', methods=['GET'])
def get_market_indicators_summary():
    try:
        indicators = format_market_indicators()
        summary_items = []
        
        if indicators['interest_rates']:
            for item in indicators['interest_rates']:
                if '콜금리' in item['name']:
                    summary_items.append({
                        'name': '콜금리',
                        'value': item['value'],
                        'change': item['change'],
                        'changePercent': item['changePercent'],
                        'trend': item['trend']
                    })
                    break
        
        if indicators['exchange_rates']:
            for item in indicators['exchange_rates']:
                if 'USD/KRW' in item['name']:
                    summary_items.append({
                        'name': 'USD/KRW',
                        'value': item['value'],
                        'change': item['change'],
                        'changePercent': item['changePercent'],
                        'trend': item['trend']
                    })
                    break
        
        if indicators['stock_indices']:
            for item in indicators['stock_indices']:
                if 'KOSPI' in item['name']:
                    summary_items.append({
                        'name': 'KOSPI',
                        'value': item['value'],
                        'change': item['change'],
                        'changePercent': item['changePercent'],
                        'trend': item['trend']
                    })
                    break
        
        return jsonify({
            "status": "success",
            "data": summary_items[:6],
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"시장지표 요약 조회 중 오류 발생: {str(e)}"
        }), 500

@app.route('/api/assets', methods=['GET'])
def get_assets_endpoint():
    try:
        asset_list = list(etf_master_collection.find({}, {'_id': 0}))
        
        if not asset_list:
            return jsonify({"status": "error", "message": "ETF 데이터가 없습니다."}), 404
        
        return jsonify(asset_list), 200
    except Exception as e:
        return jsonify({"status": "error", "message": f"자산 목록 조회 중 오류: {str(e)}"}), 500

@app.route('/api/etf/<ticker>/price', methods=['GET'])
def get_etf_price_endpoint(ticker):
    try:
        if not ticker.endswith('.KS'):
            ticker = f"{ticker}.KS"
        
        price_info = get_etf_price_info(ticker)
        
        if price_info is None:
            return jsonify({"status": "error", "message": f"{ticker}의 가격 정보를 찾을 수 없습니다."}), 404
        
        return jsonify({"status": "success", "ticker": ticker, "data": price_info}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": f"가격 정보 조회 중 오류 발생: {str(e)}"}), 500

@app.route('/api/etf/<ticker>/info', methods=['GET'])
def get_etf_detail_info(ticker):
    try:
        etf_info = etf_master_collection.find_one({
            '$or': [
                {'단축코드': ticker},
                {'ticker': ticker},
                {'ticker': f"{ticker}.KS"}
            ]
        }, {'_id': 0})
        
        if not etf_info:
            return jsonify({"status": "error", "message": f"{ticker}에 해당하는 ETF 정보를 찾을 수 없습니다."}), 404
        
        etf_ticker = etf_info.get('ticker', f"{ticker}.KS")
        price_info = get_etf_price_info(etf_ticker)
        
        return jsonify({
            "status": "success",
            "basic_info": etf_info,
            "price_info": price_info
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": f"ETF 정보 조회 중 오류 발생: {str(e)}"}), 500

@app.route('/api/optimize', methods=['POST'])
def optimize_endpoint():
    data = request.get_json()
    
    if not data or "optimization_params" not in data:
        return jsonify({"status": "error", "message": "'optimization_params'가 필요합니다."}), 400
    
    params = data.get("optimization_params")
    mode = params.get("mode", "MVO")
    
    # Beginner 모드는 asset_pairs 없이도 동작
    if mode != "Beginner" and "asset_pairs" not in data:
        return jsonify({"status": "error", "message": "'asset_pairs'가 필요합니다."}), 400
    
    asset_pairs = data.get("asset_pairs", [])
    
    print("\n" + "="*60)
    print(f"🔥 받은 요청 데이터:")
    print(f"  - mode: {mode}")
    if mode == "Beginner":
        print(f"  - style_index: {params.get('style_index')}")
        print(f"  - risk_index: {params.get('risk_index')}")
    else:
        print(f"  - asset_pairs: {asset_pairs}")
        print(f"  - optimization_params: {params}")
    print("="*60 + "\n")
    
    try:
        selected_tickers, weights, performance = perform_portfolio_optimization(asset_pairs, params)
        
        # ETF 상세 정보 추가
        etf_details = []
        for ticker in selected_tickers:
            etf_info = etf_master_collection.find_one({'ticker': ticker}, {'_id': 0})
            if etf_info:
                etf_details.append({
                    'ticker': ticker,
                    'name': etf_info.get('한글종목약명', ''),
                    'code': etf_info.get('code', ''),
                    'saa_class': etf_info.get('saa_class', ''),
                    'taa_class': etf_info.get('taa_class', '')
                })
        
        result = {
            "selected_etfs": selected_tickers,
            "etf_details": etf_details,
            "weights": [{"ticker": t, "weight": f"{w*100:.2f}%"} for t, w in weights.items()],
            "performance": performance
        }
        
        # Beginner 모드는 backtesting 제외
        if mode != "Beginner":
            backtesting_results = backtester.run_backtest(weights)
            result["backtesting"] = backtesting_results
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({"status": "error", "message": f"포트폴리오 최적화 중 오류 발생: {str(e)}"}), 500

@app.route('/api/risk-analysis', methods=['POST'])
def calculate_comprehensive_risk_endpoint():
    data = request.get_json()
    
    if not data or "performance" not in data:
        return jsonify({"status": "error", "message": "'performance' 데이터가 필요합니다."}), 400
            
    performance = data.get("performance")
    risk_free_rate = data.get("risk_free_rate", 0.02)

    try:
        annual_return = performance.get('expected_annual_return')
        annual_vol = performance.get('annual_volatility')
        
        if annual_return is None or annual_vol is None:
            return jsonify({
                "status": "error", 
                "message": "performance 데이터에 'expected_annual_return'과 'annual_volatility'가 필요합니다."
            }), 400
        
        var_results = optimizer.ValueAtRisk(annual_return, annual_vol, risk_free_rate)
        shortfall_results = optimizer.shortfallrisk(annual_return, annual_vol, risk_free_rate)
        
        result = {
            "value_at_risk": var_results,
            "shortfall_risk": shortfall_results
        }
        
        return jsonify(result), 200
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"종합 리스크 분석 중 서버 오류 발생: {e}"}), 500

@app.route('/api/market-rankings/<category>', methods=['GET'])
def get_market_rankings(category):
    """
    마켓 서머리 - 자산군/ETF 등락률 순위
    
    Parameters:
    - category: 'asset' (자산군) 또는 'etf'
    - timeframe: 쿼리 파라미터 (당일, 1주일, 1달, 3개월, 6개월, 1년, 3년)
    
    Returns:
    {
        "status": "success",
        "category": "asset" | "etf",
        "timeframe": "1달",
        "top": [...],  # 상위 10개
        "bottom": [...]  # 하위 10개
    }
    """
    try:
        # 쿼리 파라미터에서 timeframe 가져오기 (기본값: 1달)
        timeframe = request.args.get('timeframe', '1달')
        
        # 유효한 타임프레임인지 확인
        valid_timeframes = ['당일', '1주일', '1달', '3개월', '6개월', '1년', '3년']
        if timeframe not in valid_timeframes:
            return jsonify({
                "status": "error",
                "message": f"유효하지 않은 타임프레임입니다. 가능한 값: {', '.join(valid_timeframes)}"
            }), 400
        
        # 카테고리별 순위 계산
        if category == 'asset':
            result = market_ranking.get_asset_class_rankings(timeframe)
        elif category == 'etf':
            result = market_ranking.get_etf_rankings(timeframe)
        else:
            return jsonify({
                "status": "error",
                "message": "유효하지 않은 카테고리입니다. 'asset' 또는 'etf'를 사용하세요."
            }), 400
        
        if result['status'] == 'error':
            return jsonify(result), 500
        
        # 카테고리 정보 추가
        result['category'] = category
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"❌ 마켓 순위 조회 오류: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"마켓 순위 조회 중 오류 발생: {str(e)}"
        }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Flask 서버 시작 (순수 MongoDB 버전 - 파일 저장 없음)")
    print("=" * 60)
    
    try:
        print("\n📡 MongoDB 연결 테스트...")
        
        etf_count = etf_master_collection.count_documents({})
        print(f"  ✅ ETF 마스터 데이터: {etf_count}개")
        
        fund_count = fund_prices_collection.count_documents({})
        print(f"  ✅ ETF 가격 데이터: {fund_count}개")
        
        ecos_count = ecos_prices_collection.count_documents({})
        print(f"  ✅ ECOS 경제지표: {ecos_count}개")
        
        print("\n🚀 Flask 서버 시작...")
        print(f"📍 포트: 8000")
        print(f"💾 로컬 파일 저장: ❌ (메모리에서만 처리)")
        print("=" * 60 + "\n")
        
        app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)
    except Exception as e:
        print(f"\n❌ 서버 시작 실패: {e}")
        import traceback
        traceback.print_exc()