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
    # 경로를 .parquet에서 .csv로 수정
    file_path = os.path.join(PRICE_DATA_DIR, f"{ticker}.csv")
    try:
        return os.path.getsize(file_path)
    except FileNotFoundError:
        return 0

@app.route('/api/assets', methods=['GET'])
def get_assets_endpoint():
    """프론트엔드에 보여줄 자산(ETF) 목록 전체를 반환합니다."""
    if not os.path.exists(MASTER_FILE_PATH):
        return jsonify({"status": "error", "message": "ETF 마스터 파일이 없습니다."}), 404
    with open(MASTER_FILE_PATH, 'r', encoding='utf-8') as f:
        asset_list = json.load(f)
    return jsonify(asset_list), 200

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
        
        etf_df.rename(columns={
            'saaclass': 'saa_class',
            'taaclass': 'taa_class'
        }, inplace=True)
        
        final_tickers = set()
        
        for pair in asset_pairs:
            saa = pair.get('saa_class') or pair.get('saaclass')
            taa = pair.get('taa_class') or pair.get('taaclass')
            
            if not saa or not taa:
                continue

            candidates = etf_df[
                (etf_df['saa_class'] == saa) & 
                (etf_df['taa_class'] == taa)
            ].copy()
            
            if not candidates.empty:
                candidates['file_size'] = candidates['ticker'].apply(get_file_size)
                top_etf = candidates.loc[candidates['file_size'].idxmax()]
                final_tickers.add(top_etf['ticker'])
                print(f"조합 ['{saa}' - '{taa}'] 대표 ETF: {top_etf.get('한글종목약명', '이름없음')}")
            else:
                print(f"경고: 조합 ['{saa}' - '{taa}']에 해당하는 ETF가 없습니다.")

        selected_tickers = sorted(list(final_tickers))
        print(f"선택된 최종 티커 목록: {selected_tickers}")

        if len(selected_tickers) < 2:
            return jsonify({"status": "error", "message": "유효한 대표 ETF를 2개 이상 선택할 수 없습니다."}), 400

        # 2. 포트폴리오 최적화 실행
        weights, performance = optimizer.get_optimized_portfolio(selected_tickers, params)
        
        # ★★★ 백테스팅 실행 로직 추가 ★★★
        backtesting_results = backtester.run_backtest(weights)
        
        result = {
            "selected_etfs": selected_tickers,
            "weights": [{"ticker": t, "weight": f"{w*100:.2f}%"} for t, w in weights.items()],
            
            # ★★★ 수정: performance 객체를 가공하지 않고 그대로 전달합니다. ★★★
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
    app.run(host='0.0.0.0', port=5000, debug=True)