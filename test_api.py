import requests
import json

API_URL = "http://127.0.0.1:5000/api/optimize"

# ==========================================================
# ★★★ 데이터 기간이 충분히 겹칠만한 안정적인 조합으로 변경 ★★★
# ==========================================================
portfolio_request = {
    "asset_pairs": [
        # 가장 기본적인 국내 대표 주식
        {"saa_class": "국내주식", "taa_class": "바이오"},
        # 가장 기본적인 해외 대표 주식
        {"saa_class": "대체투자", "taa_class": "금리"},
        # 가장 기본적인 국내 대표 채권
        {"saa_class": "국내채권", "taa_class": "회사챈"}
    ],
    "optimization_params": {
        "mode": "MVO", 
        "mvo_objective": "max_sharpe"
        # 필요한 경우 다른 파라미터 추가 가능
        # "risk_free_rate": 0.02,
        # "risk_asset_limit": 0.8
    }
}

try:
    print("="*50)
    print(f"서버에 보낼 요청 데이터:\n{json.dumps(portfolio_request, indent=4, ensure_ascii=False)}")
    print("="*50)
    
    response = requests.post(API_URL, json=portfolio_request)
    response.raise_for_status() 
    
    print("\n[성공] 서버로부터 받은 최적 포트폴리오:")
    print(json.dumps(response.json(), indent=4, ensure_ascii=False))

except requests.exceptions.RequestException as e:
    print(f"\n[오류] API 서버에 연결할 수 없거나 요청에 실패했습니다: {e}")
    if e.response:
        try:
            print(f"서버 응답: {e.response.json().get('message', e.response.text)}")
        except json.JSONDecodeError:
            print(f"서버 응답 (Raw): {e.response.text}")