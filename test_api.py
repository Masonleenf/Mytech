import requests
import json
from datetime import datetime
import time

# 백엔드 서버 URL (실제 배포된 서버 URL로 변경)
BASE_URL = "http://127.0.0.1:8000"
# 로컬 테스트시 사용: BASE_URL = "http://127.0.0.1:8000"

def print_separator(title):
    """예쁜 구분선 출력"""
    print("\n" + "="*60)
    print(f"🔥 {title}")
    print("="*60)

def print_response_details(response, api_name):
    """응답 세부사항을 자세히 출력"""
    print(f"\n📡 {api_name} API 응답:")
    print(f"   ✅ 상태코드: {response.status_code}")
    print(f"   📝 Content-Type: {response.headers.get('content-type', '알수없음')}")
    
    if response.status_code == 200:
        try:
            data = response.json()
            print(f"   📊 응답 데이터 타입: {type(data).__name__}")
            
            # 응답 구조 분석
            if isinstance(data, dict):
                print(f"   🔑 딕셔너리 키들: {list(data.keys())}")
                for key, value in data.items():
                    print(f"      - {key}: {type(value).__name__}")
                    if isinstance(value, list) and len(value) > 0:
                        print(f"        (리스트 길이: {len(value)}, 첫번째 요소 타입: {type(value[0]).__name__})")
                    elif isinstance(value, (int, float, str)):
                        print(f"        (값: {value})")
            elif isinstance(data, list):
                print(f"   📝 리스트 길이: {len(data)}")
                if len(data) > 0:
                    print(f"   🧩 첫번째 요소 타입: {type(data[0]).__name__}")
                    if isinstance(data[0], dict):
                        print(f"   🔑 첫번째 요소의 키들: {list(data[0].keys())}")
            
            # 응답 데이터 출력 (처음 부분만)
            print(f"\n📄 응답 데이터 (JSON):")
            response_str = json.dumps(data, indent=2, ensure_ascii=False)
            if len(response_str) > 1000:
                print(response_str[:1000] + "...")
            else:
                print(response_str)
            
        except json.JSONDecodeError:
            print(f"   ❌ JSON 파싱 실패, 원시 응답: {response.text[:200]}...")
    else:
        print(f"   ❌ 에러 응답: {response.text[:300]}")

# =============================================================================
# 1. 서버 상태 확인 테스트
# =============================================================================
def test_health_check():
    """서버 상태 확인 - 가장 기본적인 테스트"""
    print_separator("1. 서버 상태 확인 테스트")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print_response_details(response, "Health Check")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n🎯 분석:")
            print(f"   - 서버 상태: {data.get('status', '알수없음')}")
            print(f"   - 메시지: {data.get('message', '메시지없음')}")
            return True
        else:
            print("❌ 서버 응답 실패")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 서버 연결 실패: {e}")
        return False

# =============================================================================
# 2. 시장 지표 관련 테스트
# =============================================================================
def test_market_indicators():
    """전체 시장 지표 조회 테스트"""
    print_separator("2. 전체 시장 지표 조회 테스트")
    
    try:
        response = requests.get(f"{BASE_URL}/api/market-indicators", timeout=15)
        print_response_details(response, "Market Indicators")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n🎯 분석:")
            print(f"   - 응답 상태: {data.get('status')}")
            print(f"   - 타임스탬프: {data.get('timestamp')}")
            
            if 'data' in data:
                market_data = data['data']
                print(f"   - 시장 지표 카테고리: {list(market_data.keys())}")
                
                for category, indicators in market_data.items():
                    if isinstance(indicators, list):
                        print(f"     📊 {category}: {len(indicators)}개 지표")
                        if len(indicators) > 0:
                            first_indicator = indicators[0]
                            if isinstance(first_indicator, dict):
                                print(f"        예시 지표 구조: {list(first_indicator.keys())}")
                                print(f"        이름: {first_indicator.get('name', 'N/A')}")
                                print(f"        값: {first_indicator.get('value', 'N/A')}")
                                print(f"        변화: {first_indicator.get('change', 'N/A')}")
                                print(f"        변화율: {first_indicator.get('changePercent', 'N/A')}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ API 호출 실패: {e}")
        return False

def test_market_indicators_summary():
    """주요 시장 지표 요약 테스트"""
    print_separator("3. 시장 지표 요약 테스트")
    
    try:
        response = requests.get(f"{BASE_URL}/api/market-indicators/summary", timeout=10)
        print_response_details(response, "Market Indicators Summary")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n🎯 분석:")
            print(f"   - 응답 상태: {data.get('status')}")
            
            if 'data' in data and isinstance(data['data'], list):
                summary_data = data['data']
                print(f"   - 요약 지표 개수: {len(summary_data)}개")
                
                for i, indicator in enumerate(summary_data):
                    if isinstance(indicator, dict):
                        print(f"     {i+1}. {indicator.get('name', 'N/A')}: {indicator.get('value', 'N/A')} ({indicator.get('trend', 'N/A')})")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ API 호출 실패: {e}")
        return False

# =============================================================================
# 3. ETF 자산 관련 테스트
# =============================================================================
def test_assets_list():
    """ETF 자산 목록 조회 테스트"""
    print_separator("4. ETF 자산 목록 조회 테스트")
    
    try:
        response = requests.get(f"{BASE_URL}/api/assets", timeout=15)
        print_response_details(response, "Assets List")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n🎯 분석:")
            print(f"   - 응답 데이터 타입: {type(data).__name__}")
            
            if isinstance(data, list):
                print(f"   - 총 ETF 개수: {len(data)}개")
                
                if len(data) > 0:
                    first_etf = data[0]
                    if isinstance(first_etf, dict):
                        print(f"   - ETF 객체 구조: {list(first_etf.keys())}")
                        print(f"   - 첫번째 ETF 정보:")
                        for key, value in first_etf.items():
                            print(f"     * {key}: {type(value).__name__} = {value}")
                        
                        # 몇 개 더 샘플 출력
                        print(f"\n   - ETF 샘플 (처음 3개):")
                        for i in range(min(3, len(data))):
                            etf = data[i]
                            name = etf.get('ETF명', etf.get('name', 'N/A'))
                            ticker = etf.get('ticker', etf.get('단축코드', 'N/A'))
                            print(f"     {i+1}. {name} ({ticker})")
            
            # 테스트용 ticker 찾기
            test_ticker = None
            if isinstance(data, list) and len(data) > 0:
                for etf in data[:5]:  # 처음 5개만 체크
                    ticker = etf.get('ticker', etf.get('단축코드', ''))
                    if ticker:
                        test_ticker = ticker
                        break
            
            return response.status_code == 200, test_ticker
        
        return False, None
        
    except Exception as e:
        print(f"❌ API 호출 실패: {e}")
        return False, None

def test_etf_price_info(ticker):
    """특정 ETF 가격 정보 조회 테스트"""
    print_separator(f"5. ETF 가격 정보 테스트 ({ticker})")
    
    if not ticker:
        print("❌ 테스트할 ticker가 없습니다.")
        return False
    
    try:
        # .KS 없이 시도
        response = requests.get(f"{BASE_URL}/api/etf/{ticker}/price", timeout=10)
        print_response_details(response, f"ETF Price Info ({ticker})")
        
        if response.status_code != 200:
            # .KS 추가해서 재시도
            if not ticker.endswith('.KS'):
                ticker_with_ks = f"{ticker}.KS"
                print(f"\n🔄 .KS 추가해서 재시도: {ticker_with_ks}")
                response = requests.get(f"{BASE_URL}/api/etf/{ticker_with_ks}/price", timeout=10)
                print_response_details(response, f"ETF Price Info ({ticker_with_ks})")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n🎯 분석:")
            print(f"   - 응답 상태: {data.get('status')}")
            print(f"   - 티커: {data.get('ticker')}")
            
            if 'data' in data:
                price_data = data['data']
                print(f"   - 가격 정보 구조: {list(price_data.keys()) if isinstance(price_data, dict) else type(price_data).__name__}")
                
                if isinstance(price_data, dict):
                    print(f"   - 현재가: {price_data.get('current_price')} (타입: {type(price_data.get('current_price')).__name__})")
                    print(f"   - 전일가: {price_data.get('previous_price')} (타입: {type(price_data.get('previous_price')).__name__})")
                    print(f"   - 변화량: {price_data.get('price_change')} (타입: {type(price_data.get('price_change')).__name__})")
                    print(f"   - 변화율: {price_data.get('price_change_rate')}% (타입: {type(price_data.get('price_change_rate')).__name__})")
                    print(f"   - 마지막 업데이트: {price_data.get('last_updated')}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ API 호출 실패: {e}")
        return False

def test_etf_detail_info(ticker):
    """ETF 상세 정보 조회 테스트"""
    print_separator(f"6. ETF 상세 정보 테스트 ({ticker})")
    
    if not ticker:
        print("❌ 테스트할 ticker가 없습니다.")
        return False
    
    try:
        response = requests.get(f"{BASE_URL}/api/etf/{ticker}/info", timeout=10)
        print_response_details(response, f"ETF Detail Info ({ticker})")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n🎯 분석:")
            print(f"   - 응답 상태: {data.get('status')}")
            
            if 'basic_info' in data:
                basic_info = data['basic_info']
                print(f"   - 기본 정보 구조: {list(basic_info.keys()) if isinstance(basic_info, dict) else type(basic_info).__name__}")
                
                if isinstance(basic_info, dict):
                    print(f"     ETF명: {basic_info.get('ETF명', basic_info.get('name', 'N/A'))}")
                    print(f"     운용사: {basic_info.get('운용사', 'N/A')}")
                    print(f"     기초지수: {basic_info.get('기초지수', 'N/A')}")
            
            if 'price_info' in data:
                price_info = data['price_info']
                print(f"   - 가격 정보: {type(price_info).__name__}")
                if price_info:
                    print(f"     현재가: {price_info.get('current_price')}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ API 호출 실패: {e}")
        return False

# =============================================================================
# 4. 포트폴리오 최적화 테스트
# =============================================================================
def test_portfolio_optimization():
    """포트폴리오 최적화 테스트"""
    print_separator("7. 포트폴리오 최적화 테스트")
    
    # 테스트용 요청 데이터
    test_request = {
        "asset_pairs": [
            {"saa_class": "국내주식", "taa_class": "지수/코스피"},    # 존재함
            {"saa_class": "국내채권", "taa_class": "단기국채"},     # 존재함
            {"saa_class": "해외주식", "taa_class": "글로벌/테마/AI"},        # 존재함
            {"saa_class": "대체투자", "taa_class": "SOC"},        # 존재함
            {"saa_class": "단기자금", "taa_class": "단기자금"}        # 존재함
        ],
        "optimization_params": {
            "method": "efficient_frontier",
            "target_return": 0.02,
            "risk_free_rate": 0.02,
            "gamma": 0.5
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/optimize", 
            json=test_request,
            timeout=30,
            headers={'Content-Type': 'application/json'}
        )
        print_response_details(response, "Portfolio Optimization")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n🎯 분석:")
            
            if 'selected_etfs' in data:
                print(f"   - 선택된 ETF: {data['selected_etfs']}")
                print(f"   - ETF 개수: {len(data['selected_etfs'])}")
            
            if 'weights' in data:
                weights = data['weights']
                print(f"   - 가중치 개수: {len(weights)}")
                print(f"   - 가중치 구조: {type(weights).__name__}")
                if isinstance(weights, list) and len(weights) > 0:
                    print(f"   - 첫번째 가중치: {weights[0]}")
                    for w in weights:
                        if isinstance(w, dict):
                            print(f"     * {w.get('ticker', 'N/A')}: {w.get('weight', 'N/A')}")
            
            if 'performance' in data:
                perf = data['performance']
                print(f"   - 성과 지표 구조: {list(perf.keys()) if isinstance(perf, dict) else type(perf).__name__}")
                if isinstance(perf, dict):
                    print(f"     예상 연간 수익률: {perf.get('expected_annual_return')} (타입: {type(perf.get('expected_annual_return')).__name__})")
                    print(f"     연간 변동성: {perf.get('annual_volatility')} (타입: {type(perf.get('annual_volatility')).__name__})")
                    print(f"     샤프 비율: {perf.get('sharpe_ratio')} (타입: {type(perf.get('sharpe_ratio')).__name__})")
            
            if 'backtesting' in data:
                backtest = data['backtesting']
                print(f"   - 백테스팅 구조: {list(backtest.keys()) if isinstance(backtest, dict) else type(backtest).__name__}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ API 호출 실패: {e}")
        return False

def test_risk_analysis():
    """리스크 분석 테스트"""
    print_separator("8. 리스크 분석 테스트")
    
    # ✅ 먼저 포트폴리오 최적화를 실행해서 performance를 얻음
    optimize_request = {
        "asset_pairs": [
            {"saa_class": "국내주식", "taa_class": "지수/코스피"},
            {"saa_class": "국내채권", "taa_class": "단기국채"}
        ],
        "optimization_params": {
            "mode": "MVO",
            "mvo_objective": "max_sharpe",
            "target_return": 0.02,
            "risk_free_rate": 0.02
        }
    }
    
    try:
        # 1. 먼저 최적화 실행
        optimize_response = requests.post(
            f"{BASE_URL}/api/optimize", 
            json=optimize_request,
            timeout=30,
            headers={'Content-Type': 'application/json'}
        )
        
        if optimize_response.status_code != 200:
            print(f"최적화 실패: {optimize_response.text}")
            return False
        
        optimize_data = optimize_response.json()
        
        # 2. performance 데이터 추출
        if 'performance' not in optimize_data:
            print("최적화 결과에 performance가 없습니다.")
            return False
        
        # 3. 리스크 분석 요청 (performance만 전달)
        risk_request = {
            "performance": optimize_data['performance'],
            "risk_free_rate": 0.02
        }
        
        response = requests.post(
            f"{BASE_URL}/api/risk-analysis", 
            json=risk_request,
            timeout=30,
            headers={'Content-Type': 'application/json'}
        )
        
        print_response_details(response, "Risk Analysis")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n🎯 분석:")
            
            if 'value_at_risk' in data:
                var_data = data['value_at_risk']
                print(f"   - VaR 분석 구조: {list(var_data.keys()) if isinstance(var_data, dict) else type(var_data).__name__}")
            
            if 'shortfall_risk' in data:
                shortfall_data = data['shortfall_risk']
                print(f"   - Shortfall Risk 구조: {list(shortfall_data.keys()) if isinstance(shortfall_data, dict) else type(shortfall_data).__name__}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ API 호출 실패: {e}")
        return False

# =============================================================================
# 5. 향상된 사이클 분석 테스트 (새로 추가)
# =============================================================================
def test_get_cycle_codes():
    """사이클 분석 가능 코드 목록 조회 테스트"""
    print_separator("9. 사이클 분석 코드 목록 조회 테스트")
    try:
        response = requests.get(f"{BASE_URL}/api/cycle-codes", timeout=15)
        print_response_details(response, "Cycle Codes List")
        
        test_codes = []
        if response.status_code == 200:
            data = response.json()
            print("\n🎯 분석:")
            print(f"   - 총 코드 개수: {data.get('total_codes', 'N/A')}")
            print(f"   - 지원 시간 범위: {data.get('supported_time_ranges', 'N/A')}")
            
            if 'codes' in data and isinstance(data['codes'], list) and len(data['codes']) > 0:
                codes_list = data['codes']
                print(f"   - 첫번째 코드 정보: {codes_list[0]}")
                for item in codes_list:
                    if 'code' in item:
                        test_codes.append(item['code'])
                print(f"   - 테스트용 코드 추출: {test_codes[:5]}")
                return True, test_codes
            else:
                print("   - 분석 가능한 코드가 없습니다.")
                return False, []

        return False, []
    except Exception as e:
        print(f"❌ API 호출 실패: {e}")
        return False, []

def test_enhanced_cycle_analysis(code):
    """향상된 사이클 분석 테스트 (시간 범위 지원)"""
    print_separator(f"10. 향상된 사이클 분석 테스트 ({code})")
    if not code:
        print("❌ 테스트할 코드가 없습니다.")
        return False
    
    # 여러 시간 범위로 테스트
    time_ranges = ['1개월', '3개월', '6개월', '12개월']
    success_count = 0
    
    for time_range in time_ranges:
        try:
            print(f"\n📅 시간 범위: {time_range}")
            response = requests.get(
                f"{BASE_URL}/api/cycle-analysis/{code}",
                params={'time_range': time_range},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ {time_range} 분석 성공")
                
                if 'data' in data:
                    analysis_data = data['data']
                    
                    # HAC 추정 사용 여부 확인
                    hac_used = analysis_data.get('analysis_summary', {}).get('hac_estimation_used', False)
                    print(f"   - HAC 추정 사용: {hac_used}")
                    
                    # 데이터 구조 확인
                    if 'historical_data' in analysis_data:
                        hist_count = len(analysis_data['historical_data'])
                        print(f"   - 히스토리 데이터: {hist_count}개")
                    
                    if 'prediction_data' in analysis_data:
                        pred_count = len(analysis_data['prediction_data'])
                        print(f"   - 예측 데이터: {pred_count}개")
                        
                        # 신뢰구간 HAC 기반 여부 확인
                        if analysis_data['prediction_data']:
                            first_pred = analysis_data['prediction_data'][0]
                            has_confidence = 'upperBand' in first_pred and 'lowerBand' in first_pred
                            print(f"   - 신뢰구간 포함: {has_confidence}")
                    
                    if 'cycle_patterns' in analysis_data:
                        patterns = analysis_data['cycle_patterns']
                        print(f"   - 사이클 패턴: {list(patterns.keys())}")
                
                success_count += 1
            else:
                print(f"   ❌ {time_range} 분석 실패: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ {time_range} 테스트 오류: {e}")
    
    print(f"\n🎯 시간 범위 테스트 결과: {success_count}/{len(time_ranges)} 성공")
    return success_count > 0

def test_enhanced_cycle_table(code):
    """향상된 사이클 테이블 분석 테스트"""
    print_separator(f"11. 향상된 사이클 테이블 분석 테스트 ({code})")
    if not code:
        print("❌ 테스트할 코드가 없습니다.")
        return False
        
    try:
        response = requests.get(f"{BASE_URL}/api/cycle-table/{code}", timeout=30)
        print_response_details(response, f"Enhanced Cycle Table ({code})")

        if response.status_code == 200:
            data = response.json()
            print("\n🎯 분석:")
            if 'data' in data:
                table_data = data['data']
                
                # HAC 추정 사용 여부 확인
                hac_used = table_data.get('hac_estimation_used', False)
                print(f"   - HAC 추정 사용: {hac_used}")
                
                if 'tables' in table_data:
                    tables = table_data['tables']
                    print(f"   - 분석 테이블: {list(tables.keys())}")
                    
                    # 각 테이블의 구조 확인
                    for table_name, table_content in tables.items():
                        if isinstance(table_content, dict):
                            rows = table_content.get('rows', [])
                            headers = table_content.get('headers', [])
                            print(f"     - {table_name}: {len(headers)}개 컬럼, {len(rows)}개 행")
                            
                            # HAC 표시 확인 (성분별 분석 테이블에서)
                            if table_name == 'component_analysis' and rows:
                                for row in rows:
                                    if isinstance(row, list) and len(row) > 0:
                                        component_name = row[0]
                                        if "(HAC)" in component_name:
                                            print(f"       ✅ HAC 조정 표시 확인: {component_name}")
                
                if 'key_insights' in table_data:
                    insights = table_data['key_insights']
                    print(f"   - 주요 인사이트: {len(insights)}개")
                    
                    # HAC 기반 인사이트 확인
                    for insight in insights:
                        if "HAC" in insight:
                            print(f"     ✅ HAC 기반 인사이트: {insight[:50]}...")

        return response.status_code == 200
    except Exception as e:
        print(f"❌ API 호출 실패: {e}")
        return False

def test_frontend_compatibility(code):
    """프런트엔드 호환성 테스트"""
    print_separator(f"12. 프런트엔드 호환성 테스트 ({code})")
    if not code:
        print("❌ 테스트할 코드가 없습니다.")
        return False
    
    try:
        # 차트 데이터 호환성 테스트
        response = requests.get(
            f"{BASE_URL}/api/cycle-analysis/{code}/chart-data",
            params={'time_range': '3개월'},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 차트 데이터 API 응답 성공")
            
            if 'chart_data' in data:
                chart_data = data['chart_data']
                
                if isinstance(chart_data, list) and len(chart_data) > 0:
                    first_item = chart_data[0]
                    
                    # CycleChart.tsx 필수 필드 확인
                    required_fields = ['date', 'combined', 'trend', 'businessCycle', 
                                     'shortCycle', 'noise', 'isPrediction']
                    
                    missing_fields = []
                    for field in required_fields:
                        if field not in first_item:
                            missing_fields.append(field)
                    
                    if not missing_fields:
                        print("   ✅ 모든 필수 필드 존재")
                        
                        # 날짜 형식 확인
                        try:
                            datetime.fromisoformat(first_item['date'].replace('Z', '+00:00'))
                            print("   ✅ 날짜 형식 올바름 (ISO format)")
                        except:
                            print("   ❌ 날짜 형식 오류")
                        
                        # 예측 데이터 확인
                        prediction_items = [item for item in chart_data if item.get('isPrediction', False)]
                        print(f"   - 예측 데이터 개수: {len(prediction_items)}")
                        
                        # 신뢰구간 데이터 확인
                        if prediction_items:
                            has_bands = all('upperBand' in item and 'lowerBand' in item 
                                          for item in prediction_items)
                            print(f"   - 신뢰구간 포함: {has_bands}")
                        
                    else:
                        print(f"   ❌ 누락 필드: {missing_fields}")
                        return False
                
            # 사이클 위치 데이터 확인
            if 'cycle_positions' in data:
                cycle_pos = data['cycle_positions']
                expected_cycles = ['business_cycle', 'short_cycle']
                
                for cycle_type in expected_cycles:
                    if cycle_type in cycle_pos:
                        cycle_info = cycle_pos[cycle_type]
                        required_cycle_fields = ['position', 'trend', 'current_value', 'color']
                        
                        cycle_missing = [f for f in required_cycle_fields if f not in cycle_info]
                        if not cycle_missing:
                            print(f"   ✅ {cycle_type} 위치 데이터 완전")
                        else:
                            print(f"   ❌ {cycle_type} 누락 필드: {cycle_missing}")
            
            return True
        else:
            print(f"❌ 차트 데이터 API 실패: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 프런트엔드 호환성 테스트 실패: {e}")
        return False

def test_pattern_wheel_data(code):
    """패턴 휠 데이터 테스트 (PatternScreen.tsx용)"""
    print_separator(f"13. 패턴 휠 데이터 테스트 ({code})")
    if not code:
        print("❌ 테스트할 코드가 없습니다.")
        return False
    
    try:
        response = requests.get(f"{BASE_URL}/api/cycle-analysis/{code}/pattern-data", timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 패턴 휠 데이터 API 응답 성공")
            
            if 'pattern_data' in data:
                pattern_data = data['pattern_data']
                
                if isinstance(pattern_data, list):
                    print(f"   - 패턴 개수: {len(pattern_data)}")
                    
                    required_pattern_fields = ['type', 'currentPosition', 'recentTrend', 
                                             'averagePeriod', 'color', 'position_value']
                    
                    for i, pattern in enumerate(pattern_data):
                        if isinstance(pattern, dict):
                            missing = [f for f in required_pattern_fields if f not in pattern]
                            if not missing:
                                print(f"   ✅ 패턴 {i+1} ({pattern.get('type')}): 완전")
                                print(f"     - 위치: {pattern.get('currentPosition')}")
                                print(f"     - 추세: {pattern.get('recentTrend')}")
                                print(f"     - 주기: {pattern.get('averagePeriod')}")
                                print(f"     - 위치값: {pattern.get('position_value')}")
                            else:
                                print(f"   ❌ 패턴 {i+1} 누락 필드: {missing}")
                                return False
            
            return True
        else:
            print(f"❌ 패턴 휠 데이터 API 실패: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 패턴 휠 데이터 테스트 실패: {e}")
        return False

def test_cycle_quick_analysis(codes):
    """여러 코드에 대한 빠른 사이클 분석 테스트"""
    print_separator(f"14. 사이클 요약 분석 테스트 ({codes})")
    if not codes:
        print("❌ 테스트할 코드가 없습니다.")
        return False

    test_request = {"codes": codes}
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/cycle-quick-analysis",
            json=test_request,
            timeout=30,
            headers={'Content-Type': 'application/json'}
        )
        print_response_details(response, "Cycle Quick Analysis")

        if response.status_code == 200:
            data = response.json()
            print("\n🎯 분석:")
            print(f"   - 요청 코드 수: {data.get('requested_codes')}")
            print(f"   - 성공 분석 수: {data.get('successful_analyses')}")
            if 'results' in data and len(data['results']) > 0:
                print(f"   - 첫번째 결과: {data['results'][0]}")

        return response.status_code == 200
    except Exception as e:
        print(f"❌ API 호출 실패: {e}")
        return False

def test_cycle_spectrum(code):
    """특정 코드 주파수 스펙트럼 분석 테스트"""
    print_separator(f"15. 사이클 스펙트럼 분석 테스트 ({code})")
    if not code:
        print("❌ 테스트할 코드가 없습니다.")
        return False
        
    try:
        response = requests.get(f"{BASE_URL}/api/cycle-spectrum/{code}", timeout=30)
        print_response_details(response, f"Cycle Spectrum ({code})")

        if response.status_code == 200:
            data = response.json().get('data', {})
            print("\n🎯 분석:")
            if 'dominant_cycles' in data and data.get('dominant_cycles'):
                top_cycle = data['dominant_cycles'][0]
                print(f"   - 주요 사이클 개수: {len(data['dominant_cycles'])}개")
                print(f"   - 가장 강한 주기: 약 {top_cycle.get('period_days', 'N/A'):.1f}일")
                print(f"     (신호 강도: {top_cycle.get('strength', 'N/A'):.2f})")
            if 'periods' in data:
                print(f"   - 분석된 주기 포인트: {len(data['periods'])}개")

        return response.status_code == 200
    except Exception as e:
        print(f"❌ API 호출 실패: {e}")
        return False

# =============================================================================
# 메인 테스트 실행 함수
# =============================================================================
def main():
    """모든 API 테스트를 순차적으로 실행 (향상된 버전)"""
    print(f"🚀 완전한 API 테스트 시작 (향상된 사이클 분석 포함)")
    print(f"🌐 대상 서버: {BASE_URL}")
    print(f"🕐 테스트 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # 1. 서버 상태 확인 (필수)
    print("\n" + "🔥 Step 1: 서버 연결 상태 확인")
    server_ok = test_health_check()
    results['서버 상태'] = server_ok
    
    if not server_ok:
        print("\n❌ 서버 연결 실패로 테스트를 중단합니다.")
        return
    
    # 2. 시장 지표 테스트
    print("\n" + "🔥 Step 2: 시장 지표 API 테스트")
    results['시장 지표 전체'] = test_market_indicators()
    results['시장 지표 요약'] = test_market_indicators_summary()
    
    # 3. ETF 자산 관련 테스트
    print("\n" + "🔥 Step 3: ETF 자산 API 테스트")
    assets_ok, test_ticker = test_assets_list()
    results['ETF 자산 목록'] = assets_ok
    
    if test_ticker:
        results['ETF 가격 정보'] = test_etf_price_info(test_ticker)
        results['ETF 상세 정보'] = test_etf_detail_info(test_ticker)
    else:
        print("⚠️ 테스트할 ticker를 찾지 못해 ETF 개별 정보 테스트를 건너뜁니다.")
        results['ETF 가격 정보'] = False
        results['ETF 상세 정보'] = False
    
    # 4. 포트폴리오 최적화 테스트
    print("\n" + "🔥 Step 4: 포트폴리오 기능 테스트")
    results['포트폴리오 최적화'] = test_portfolio_optimization()
    results['리스크 분석'] = test_risk_analysis()
    
    # 5. 향상된 사이클 분석 테스트
    print("\n" + "🔥 Step 5: 향상된 사이클 분석 기능 테스트")
    cycle_codes_ok, test_cycle_code_list = test_get_cycle_codes()
    results['사이클 코드 목록'] = cycle_codes_ok
    
    if test_cycle_code_list:
        test_code = test_cycle_code_list[0]  # 첫 번째 사용 가능한 코드 사용
        
        # 향상된 사이클 분석 테스트들
        results['향상된 사이클 분석'] = test_enhanced_cycle_analysis(test_code)
        results['향상된 사이클 테이블'] = test_enhanced_cycle_table(test_code)
        results['프런트엔드 호환성'] = test_frontend_compatibility(test_code)
        results['패턴 휠 데이터'] = test_pattern_wheel_data(test_code)
        results['사이클 요약 분석'] = test_cycle_quick_analysis(test_cycle_code_list[:3])
        results['사이클 스펙트럼 분석'] = test_cycle_spectrum(test_code)
    else:
        print("⚠️ 테스트할 사이클 분석 코드를 찾지 못해 관련 테스트를 건너뜁니다.")
        results['향상된 사이클 분석'] = False
        results['향상된 사이클 테이블'] = False
        results['프런트엔드 호환성'] = False
        results['패턴 휠 데이터'] = False
        results['사이클 요약 분석'] = False
        results['사이클 스펙트럼 분석'] = False

    # 최종 결과 출력
    print_separator("🎯 최종 테스트 결과 요약")
    print(f"📊 테스트 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    
    print(f"\n✅ 성공: {passed_tests}/{total_tests}")
    print(f"❌ 실패: {total_tests - passed_tests}/{total_tests}")
    
    print(f"\n📋 상세 결과:")
    for test_name, result in results.items():
        status = "✅ 성공" if result else "❌ 실패"
        print(f"   {status} {test_name}")
    
    # 성공률 계산
    success_rate = (passed_tests / total_tests) * 100
    print(f"\n🎯 전체 성공률: {success_rate:.1f}%")
    
    # 향상된 기능 테스트 결과 별도 분석
    enhanced_tests = ['향상된 사이클 분석', '향상된 사이클 테이블', '프런트엔드 호환성', '패턴 휠 데이터']
    enhanced_passed = sum(1 for test_name in enhanced_tests if results.get(test_name, False))
    
    print(f"\n🔬 향상된 기능 테스트:")
    print(f"   HAC 추정 및 시간 범위 지원: {enhanced_passed}/{len(enhanced_tests)} 성공")
    
    if success_rate == 100:
        print("🎉 모든 API가 정상적으로 작동합니다!")
        if enhanced_passed == len(enhanced_tests):
            print("🎊 향상된 기능들도 모두 정상 작동합니다!")
    elif success_rate >= 80:
        print("👍 대부분의 API가 정상적으로 작동합니다.")
    else:
        print("⚠️ 일부 API에 문제가 있을 수 있습니다. 로그를 확인해주세요.")

if __name__ == "__main__":
    main()