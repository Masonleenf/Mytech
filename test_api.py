import requests
import json

# 백엔드 서버 URL
BASE_URL = "http://127.0.0.1:5000"

def test_price_with_padding():
    """6자리 패딩된 형식으로 가격 정보 테스트"""
    test_cases = [
        ("117680", "117680.KS.csv"),
    ]
    
    for original_ticker, expected_file in test_cases:
        print(f"\n--- {original_ticker} 테스트 (예상 파일: {expected_file}) ---")
        
        # 1. 원본 형식으로 시도
        test_price_info(original_ticker)
        
        # 2. .KS 추가해서 시도  
        test_price_info(f"{original_ticker}.KS")

def test_price_info(ticker):
    """특정 ETF 가격 정보 조회"""
    try:
        # ✅ 올바른 API 엔드포인트 사용
        url = f"{BASE_URL}/api/etf/{ticker}/price"
        print(f"Price API 요청: {url}")
        response = requests.get(url)
        print(f"Price API: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 성공: {json.dumps(data, indent=2, ensure_ascii=False)}")
        else:
            print(f"❌ 실패 ({response.status_code}): {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ API 호출 실패: {e}")

def test_etf_detail_info(ticker):
    """ETF 상세 정보 조회 테스트"""
    try:
        url = f"{BASE_URL}/api/etf/{ticker}/info"
        print(f"Detail API 요청: {url}")
        response = requests.get(url)
        print(f"Detail API: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 성공: {json.dumps(data, indent=2, ensure_ascii=False)}")
        else:
            print(f"❌ 실패 ({response.status_code}): {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ API 호출 실패: {e}")

def test_assets_list():
    """전체 자산 목록 조회 테스트"""
    try:
        url = f"{BASE_URL}/api/assets"
        print(f"Assets API 요청: {url}")
        response = requests.get(url)
        print(f"Assets API: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 성공: 총 {len(data)}개 자산")
            if data:
                print(f"첫 번째 자산 예시: {json.dumps(data[0], indent=2, ensure_ascii=False)}")
        else:
            print(f"❌ 실패 ({response.status_code}): {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ API 호출 실패: {e}")

def main():
    print("=== Flask API 엔드포인트 테스트 ===")
    
    # 서버 상태 먼저 확인
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("서버가 정상 작동하지 않습니다.")
            return
        print("✅ 서버 정상 작동 확인")
    except:
        print("❌ 서버 연결 실패")
        return
    
    # 1. 전체 자산 목록 테스트
    print("\n" + "="*50)
    print("1. 전체 자산 목록 조회 테스트")
    test_assets_list()
    
    # 2. 가격 정보 테스트
    print("\n" + "="*50)
    print("2. 가격 정보 조회 테스트")
    test_price_with_padding()
    
    # 3. 상세 정보 테스트
    print("\n" + "="*50)
    print("3. 상세 정보 조회 테스트")
    test_etf_detail_info("117680")

if __name__ == "__main__":
    main()