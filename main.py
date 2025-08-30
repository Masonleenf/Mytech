from flask import Flask, jsonify, request
from flask_cors import CORS
import optimizer
import backtester
import json
import os
import pandas as pd
from datetime import datetime

app = Flask(__name__)
CORS(app)

DATA_DIR = "data"
PRICE_DATA_DIR = os.path.join(DATA_DIR, "fund_prices")
MASTER_FILE_PATH = os.path.join(DATA_DIR, "etf_master.json")
ECOS_DATA_PATH = os.path.join(DATA_DIR, "ecos_data.json")  # ECOS 데이터 경로 추가

def load_ecos_data():
    """ECOS 데이터 로드"""
    try:
        if not os.path.exists(ECOS_DATA_PATH):
            print(f"❌ ECOS 데이터 파일이 없습니다: {ECOS_DATA_PATH}")
            return {}
        
        with open(ECOS_DATA_PATH, 'r', encoding='utf-8') as f:
            ecos_data = json.load(f)
        
        print(f"✅ ECOS 데이터 로드 완료: {len(ecos_data)}개 통계")
        return ecos_data
    
    except Exception as e:
        print(f"❌ ECOS 데이터 로드 실패: {e}")
        return {}

def get_latest_market_data(stat_code, item_code1, ecos_data):
    """특정 통계의 최신 2일 데이터 조회"""
    try:
        key = f"{stat_code}_{item_code1}"
        
        if key not in ecos_data or not ecos_data[key]['data']:
            return None
        
        data_list = ecos_data[key]['data']
        stat_info = ecos_data[key]['info']
        
        # 날짜순으로 정렬하고 최신 2개 데이터 가져오기
        sorted_data = sorted(data_list, key=lambda x: x['TIME'])
        
        if len(sorted_data) < 1:
            return None
        
        latest = sorted_data[-1]
        previous = sorted_data[-2] if len(sorted_data) >= 2 else None
        
        current_value = float(latest['DATA_VALUE'])
        previous_value = float(previous['DATA_VALUE']) if previous else current_value
        
        # 변화량 및 변화율 계산
        change = current_value - previous_value
        change_percent = (change / previous_value * 100) if previous_value != 0 else 0
        
        # 트렌드 결정
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
        print(f"시장 데이터 조회 오류 ({stat_code}_{item_code1}): {e}")
        return None

def format_market_indicators():
    """시장지표 데이터를 프런트엔드 형식으로 변환"""
    ecos_data = load_ecos_data()
    
    if not ecos_data:
        return {
            'interest_rates': [],
            'stock_indices': [],
            'exchange_rates': []
        }
    
    # 주요 시장지표 매핑 (stat_code, item_code1)
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
            ('817Y002', 10320000, 'percent', '회사채(3년, BBB-)', 'BBB-` 회사채'),
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
            data = get_latest_market_data(stat_code, item_code1, ecos_data)
            
            if data:
                # 표시 형식 조정
                if category == 'interest_rates':
                    # 금리는 % 표시
                    value_display = f"{float(data['value']):.2f}%"
                    change_display = f"{float(data['change']):+.2f}%p"
                elif category == 'stock_indices':
                    # 주가지수는 소수점 1자리
                    value_display = f"{float(data['value']):,.1f}"
                    change_display = f"{float(data['change']):+.1f}"
                else:  # exchange_rates
                    # 환율은 소수점 2자리
                    value_display = f"{float(data['value']):,.2f}원"
                    change_display = f"{float(data['change']):+.1f}"
                
                formatted_data = {
                    'id': f"{stat_code}_{item_code1}",
                    'icon': icon_type,  # 프런트엔드에서 아이콘 매핑용
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

def get_file_size(ticker):
    """주어진 티커의 CSV 데이터 파일 크기를 확인합니다."""
    file_path = os.path.join(PRICE_DATA_DIR, f"{ticker}.csv")
    try:
        return os.path.getsize(file_path)
    except FileNotFoundError:
        return 0

def get_etf_price_info(ticker):
    """ETF의 현재가, 전일대비, 등락률 정보를 반환합니다."""
    try:
        file_path = os.path.join(PRICE_DATA_DIR, f"{ticker}.csv")
        if not os.path.exists(file_path):
            return None
        
        # CSV 파일 읽기 (헤더 3줄 스킵)
        df = pd.read_csv(
            file_path, 
            skiprows=3, 
            header=None,
            names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'],
            index_col='Date',
            parse_dates=True
        )
        
        # 데이터 정리
        df = df[df.index.notna()]
        df = df[~df.index.duplicated(keep='first')]
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna(subset=['Close'])
        
        if len(df) < 2:
            return None
        
        # 최신 2일의 데이터 가져오기
        latest_data = df.tail(2)
        
        current_price = latest_data['Close'].iloc[-1]  # 최신 종가
        previous_price = latest_data['Close'].iloc[-2]  # 전일 종가
        
        # 전일대비 계산
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
        print(f"가격 정보 조회 오류 ({ticker}): {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """서버 상태 확인 엔드포인트"""
    return jsonify({"status": "healthy", "message": "API 서버가 정상 작동 중입니다."}), 200

# ===== 시장지표 관련 API =====
@app.route('/api/market-indicators', methods=['GET'])
def get_market_indicators():
    """전체 시장지표 조회"""
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
    """카테고리별 시장지표 조회"""
    try:
        valid_categories = ['interest_rates', 'stock_indices', 'exchange_rates']
        
        if category not in valid_categories:
            return jsonify({
                "status": "error", 
                "message": f"유효하지 않은 카테고리입니다. 사용 가능: {', '.join(valid_categories)}"
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
            "message": f"{category} 시장지표 조회 중 오류 발생: {str(e)}"
        }), 500

@app.route('/api/market-indicators/summary', methods=['GET'])
def get_market_indicators_summary():
    """주요 시장지표 요약 (상단 카드용)"""
    try:
        indicators = format_market_indicators()
        
        # 주요 지표만 선별 (상단 카드용)
        summary_items = []
        
        # 콜금리
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
        
        # USD/KRW
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
        
        # KOSPI
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
        
        # 국고채 3Y
        if indicators['interest_rates']:
            for item in indicators['interest_rates']:
                if '국고채 3Y' in item['name']:
                    summary_items.append({
                        'name': '국고채3Y',
                        'value': item['value'],
                        'change': item['change'],
                        'changePercent': item['changePercent'],
                        'trend': item['trend']
                    })
                    break
        
        # KOSDAQ
        if indicators['stock_indices']:
            for item in indicators['stock_indices']:
                if 'KOSDAQ' in item['name']:
                    summary_items.append({
                        'name': 'KOSDAQ',
                        'value': item['value'],
                        'change': item['change'],
                        'changePercent': item['changePercent'],
                        'trend': item['trend']
                    })
                    break
        
        # EUR/KRW
        if indicators['exchange_rates']:
            for item in indicators['exchange_rates']:
                if 'EUR/KRW' in item['name']:
                    summary_items.append({
                        'name': 'EUR/KRW',
                        'value': item['value'],
                        'change': item['change'],
                        'changePercent': item['changePercent'],
                        'trend': item['trend']
                    })
                    break
        
        return jsonify({
            "status": "success",
            "data": summary_items[:6],  # 최대 6개
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"시장지표 요약 조회 중 오류 발생: {str(e)}"
        }), 500

# ===== 기존 ETF 관련 API =====
@app.route('/api/assets', methods=['GET'])
def get_assets_endpoint():
    """프론트엔드에 보여줄 자산(ETF) 목록 전체를 반환합니다."""
    if not os.path.exists(MASTER_FILE_PATH):
        return jsonify({"status": "error", "message": "ETF 마스터 파일이 없습니다."}), 404
    with open(MASTER_FILE_PATH, 'r', encoding='utf-8') as f:
        asset_list = json.load(f)
    return jsonify(asset_list), 200

@app.route('/api/etf/<ticker>/price', methods=['GET'])
def get_etf_price_endpoint(ticker):
    """특정 ETF의 현재가 정보를 반환합니다."""
    try:
        # 티커 정규화 (.KS 추가)
        if not ticker.endswith('.KS'):
            ticker = f"{ticker}.KS"
        
        price_info = get_etf_price_info(ticker)
        
        if price_info is None:
            return jsonify({
                "status": "error", 
                "message": f"{ticker}의 가격 정보를 찾을 수 없습니다."
            }), 404
        
        return jsonify({
            "status": "success",
            "ticker": ticker,
            "data": price_info
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"가격 정보 조회 중 오류 발생: {str(e)}"
        }), 500

@app.route('/api/etf/<ticker>/info', methods=['GET'])
def get_etf_detail_info(ticker):
    """특정 ETF의 상세 정보 (기본 정보 + 가격 정보)를 반환합니다."""
    try:
        # ETF 기본 정보 조회
        if not os.path.exists(MASTER_FILE_PATH):
            return jsonify({"status": "error", "message": "ETF 마스터 파일이 없습니다."}), 404
        
        with open(MASTER_FILE_PATH, 'r', encoding='utf-8') as f:
            etf_list = json.load(f)
        
        # 단축코드 또는 ticker로 검색
        etf_info = None
        for etf in etf_list:
            if (etf.get('단축코드') == ticker or 
                etf.get('ticker') == ticker or 
                etf.get('ticker') == f"{ticker}.KS"):
                etf_info = etf
                break
        
        if not etf_info:
            return jsonify({
                "status": "error", 
                "message": f"{ticker}에 해당하는 ETF 정보를 찾을 수 없습니다."
            }), 404
        
        # 가격 정보 조회
        etf_ticker = etf_info.get('ticker', f"{ticker}.KS")
        price_info = get_etf_price_info(etf_ticker)
        
        # 응답 데이터 구성
        response_data = {
            "status": "success",
            "basic_info": etf_info,
            "price_info": price_info
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"ETF 정보 조회 중 오류 발생: {str(e)}"
        }), 500

@app.route('/api/optimize', methods=['POST'])
def optimize_endpoint():
    """사용자로부터 자산 조합과 최적화 파라미터를 받아 포트폴리오를 계산하고 백테스팅을 수행합니다."""
    data = request.get_json()
    print("\n--- 새로운 최적화 요청 수신 ---")
    print(f"요청 데이터: {data}")
    
    if not data or "asset_pairs" not in data or "optimization_params" not in data:
        return jsonify({"status": "error", "message": "'asset_pairs'와 'optimization_params'가 모두 필요합니다."}), 400
            
    asset_pairs = data.get("asset_pairs")
    params = data.get("optimization_params")

    try:
        with open(MASTER_FILE_PATH, 'r', encoding='utf-8') as f:
            etf_df = pd.DataFrame(json.load(f))
        
        # ★★★★★ [수정된 로직 시작] ★★★★★
        # 이제 티커 대신 합성 지수 'code'를 찾습니다.
        final_codes = set()
        
        for pair in asset_pairs:
            saa = pair.get('saa_class')
            taa = pair.get('taa_class')
            
            if not saa or not taa:
                continue

            # asset pair에 해당하는 행을 찾아 'code'를 가져옵니다.
            matched_etf = etf_df[
                (etf_df['saa_class'] == saa) & 
                (etf_df['taa_class'] == taa)
            ]
            
            if not matched_etf.empty:
                code = matched_etf['code'].iloc[0]
                final_codes.add(code)
                print(f"조합 ['{saa}' - '{taa}'] 대표 코드: {code}")
            else:
                print(f"경고: 조합 ['{saa}' - '{taa}']에 해당하는 ETF가 없습니다.")

        selected_codes = sorted(list(final_codes))
        print(f"선택된 최종 코드 목록: {selected_codes}")

        if len(selected_codes) < 2:
            return jsonify({"status": "error", "message": "유효한 대표 코드를 2개 이상 선택할 수 없습니다."}), 400

        # 포트폴리오 최적화 실행 시 티커 리스트 대신 코드 리스트를 전달합니다.
        weights, performance = optimizer.get_optimized_portfolio(selected_codes, params)
        # ★★★★★ [수정된 로직 끝] ★★★★★

        # 백테스팅 실행 (백테스터는 가중치 기반이므로 수정 필요 없음)
        backtesting_results = backtester.run_backtest(weights)
        
        # 결과 반환 시 selected_etfs 대신 selected_codes를 사용
        result = {
            "selected_etfs": selected_codes,
            "weights": [{"ticker": t, "weight": f"{w*100:.2f}%"} for t, w in weights.items()],
            "performance": performance,
            "backtesting": backtesting_results
        }
        return jsonify(result), 200
        
    except (ValueError, FileNotFoundError) as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"최적화 중 서버 오류 발생: {e}"}), 500

if __name__ == '__main__':
    print("=== Flask 서버 시작 ===")
    print(f"데이터 디렉토리: {DATA_DIR}")
    print(f"ECOS 데이터 파일: {ECOS_DATA_PATH}")
    print(f"ETF 마스터 파일: {MASTER_FILE_PATH}")
    
    # 시작 시 ECOS 데이터 로드 테스트
    ecos_data = load_ecos_data()
    if ecos_data:
        print(f"✅ ECOS 데이터 준비 완료: {len(ecos_data)}개 통계")
    else:
        print("⚠️  ECOS 데이터가 없습니다. ecos_main.py를 먼저 실행하세요.")
    
    app.run(host='0.0.0.0', port=8000, debug=True)