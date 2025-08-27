from flask import Flask, jsonify, request
from flask_cors import CORS
import optimizer
import backtester
import json
import os
import pandas as pd

app = Flask(__name__)
CORS(app)

DATA_DIR = "data"
PRICE_DATA_DIR = os.path.join(DATA_DIR, "fund_prices")
MASTER_FILE_PATH = os.path.join(DATA_DIR, "etf_master.json")

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
    app.run(host='0.0.0.0', port=8000, debug=True)