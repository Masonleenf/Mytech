import requests
import pandas as pd
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

class ECOSDataManager:
    def __init__(self, api_key, data_dir="./"):
        """
        한국은행 ECOS 데이터 관리 클래스
        
        Args:
            api_key (str): 한국은행 API 키
            data_dir (str): 데이터 저장 디렉토리
        """
        self.api_key = api_key
        self.base_url = "https://ecos.bok.or.kr/api"
        self.data_dir = Path(data_dir)
        self.data_file = self.data_dir / "ecos_data.json"
        self.list_file = self.data_dir / "list.csv"
        
        # API 호출 제한 (1초당 1회)
        self.last_request_time = 0
        self.request_interval = 1.0
        
    def _rate_limit(self):
        """API 호출 속도 제한"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_interval:
            time.sleep(self.request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def _make_api_request(self, stat_code, item_code1, start_date, end_date, cycle='D', max_retries=3):
        """
        API 요청 실행
        
        Args:
            stat_code (str): 통계표코드
            item_code1 (str): 항목코드1
            start_date (str): 시작일자 (YYYYMMDD)
            end_date (str): 종료일자 (YYYYMMDD)
            cycle (str): 주기 (D=일간, M=월간, Q=분기, A=연간)
            max_retries (int): 최대 재시도 횟수
            
        Returns:
            dict: API 응답 데이터
        """
        # 주기 형식 변환 (기존 코드 그대로 유지)
        cycle_mapping = {'D': 'D', 'M': 'M', 'Q': 'QQ', 'A': 'YY'}
        api_cycle = cycle_mapping.get(cycle.upper(), 'DD')
        
        # 통계표별 항목코드 포맷팅 규칙 (기존 코드 그대로 유지)
        if stat_code == '817Y002':  # 금리
            formatted_item_code = f"0{str(item_code1).zfill(8)}"
        elif stat_code == '802Y001':  # 주가지수
            formatted_item_code = f"0{str(item_code1).zfill(6)}"  # 7자리로 포맷팅 (0001000)
        elif stat_code == '731Y001':  # 환율
            formatted_item_code = f"0{str(item_code1).zfill(6)}"  # 7자리로 포맷팅 (0000013)
        else:
            formatted_item_code = f"0{str(item_code1).zfill(6)}"
        
        url = f"{self.base_url}/StatisticSearch/{self.api_key}/json/kr/1/10000/{stat_code}/{api_cycle}/{start_date}/{end_date}/{formatted_item_code}"
        
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                
                print(f"API 요청 중: {stat_code} - {formatted_item_code} ({api_cycle}) ({start_date}~{end_date})")
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                
                data = response.json()
                
                # 에러 체크
                if 'RESULT' in data:
                    if data['RESULT']['CODE'] != '0':
                        print(f"API 에러: {data['RESULT']['MESSAGE']}")
                        if data['RESULT']['CODE'] == '200':  # 데이터 없음
                            return None
                        elif data['RESULT']['CODE'] == '602':  # 호출 제한
                            print("API 호출 제한, 대기 중...")
                            time.sleep(5)
                            continue
                        else:
                            return None
                
                # 데이터 확인
                if 'StatisticSearch' in data and 'row' in data['StatisticSearch']:
                    rows = data['StatisticSearch']['row']
                    print(f"✅ 데이터 수집 완료: {len(rows)}건")
                    return rows
                else:
                    print("❌ 데이터 없음")
                    return None
                    
            except requests.exceptions.RequestException as e:
                print(f"요청 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 지수적 백오프
                    
            except Exception as e:
                print(f"예외 발생: {e}")
                return None
                
        return None
    
    def load_statistics_list(self):
        """
        list.csv에서 통계 목록 로드
        
        Returns:
            list: 통계 정보 리스트
        """
        try:
            df = pd.read_csv(self.list_file)
            
            # 컬럼명 정리
            df.columns = df.columns.str.strip()
            
            statistics = []
            for _, row in df.iterrows():
                stat_info = {
                    'stat_code': str(row['stat_code']).strip(),
                    'item_code1': int(row['item_code1']),
                    'name': str(row['name']).strip(),
                    'period': str(row['period']).strip(),
                    'unit': str(row['단위']).strip()
                }
                statistics.append(stat_info)
            
            print(f"📋 통계 목록 로드 완료: {len(statistics)}개")
            return statistics
            
        except Exception as e:
            print(f"❌ list.csv 로드 실패: {e}")
            return []
    
    def load_existing_data(self):
        """
        기존 JSON 데이터 로드
        
        Returns:
            dict: 기존 데이터
        """
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"📂 기존 데이터 로드: {len(data)}개 통계")
                return data
            except Exception as e:
                print(f"❌ 기존 데이터 로드 실패: {e}")
                
        return {}
    
    def save_data(self, data):
        """
        데이터를 JSON 파일로 저장
        
        Args:
            data (dict): 저장할 데이터
        """
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 데이터 저장 완료: {self.data_file}")
            
        except Exception as e:
            print(f"❌ 데이터 저장 실패: {e}")
    
    def get_date_range(self, stat_code, period='D'):
        """
        통계별 적절한 시작일자 결정
        
        Args:
            stat_code (str): 통계표코드
            period (str): 주기 (D=일간, M=월간, Q=분기, A=연간)
            
        Returns:
            tuple: (시작일자, 종료일자)
        """
        # 주기에 따른 종료일자 형식
        if period == 'M':  # 월간
            end_date = datetime.now().strftime('%Y%m')
        elif period == 'Q':  # 분기
            end_date = datetime.now().strftime('%Y%m')
        elif period == 'A':  # 연간
            end_date = datetime.now().strftime('%Y')
        else:  # 일간
            end_date = datetime.now().strftime('%Y%m%d')
        
        # 통계별 시작일자 설정
        start_dates = {
            '817Y002': '19950101' if period == 'D' else '199501',  # 금리: 1995년부터
            '817Y001': '19800104' if period == 'D' else '198001',  # 주가지수: 1980년부터  
            '731Y001': '19950101' if period == 'D' else '199501',  # 환율: 1995년부터
        }
        
        default_start = '20000101' if period == 'D' else '200001'
        start_date = start_dates.get(stat_code, default_start)
        
        return start_date, end_date
    
    def get_latest_date(self, existing_data, stat_code, item_code1):
        """
        기존 데이터에서 최신 날짜 추출
        
        Args:
            existing_data (dict): 기존 데이터
            stat_code (str): 통계표코드
            item_code1 (int): 항목코드1
            
        Returns:
            str: 최신 날짜 (YYYYMMDD) 또는 None
        """
        key = f"{stat_code}_{item_code1}"
        
        if key in existing_data and existing_data[key]['data']:
            dates = [item['TIME'] for item in existing_data[key]['data']]
            latest_date = max(dates)
            return latest_date
        
        return None
    
    def print_data_structure(self, data):
        """데이터 구조 출력 (프런트엔드 연결용)"""
        if not data:
            print("📊 데이터 구조: 빈 데이터")
            return
            
        print("\n" + "="*80)
        print("📊 ECOS 데이터 구조 (프런트엔드 연결용)")
        print("="*80)
        
        # 전체 구조 요약
        print(f"📈 총 통계 개수: {len(data)}")
        
        # 통계표별 분류
        stat_groups = {}
        for key, info in data.items():
            stat_code = info['info']['stat_code']
            if stat_code not in stat_groups:
                stat_groups[stat_code] = []
            stat_groups[stat_code].append(info)
        
        print(f"📋 통계표 개수: {len(stat_groups)}")
        for stat_code, items in stat_groups.items():
            print(f"  - {stat_code}: {len(items)}개 항목")
        
        # 샘플 데이터 구조 출력
        sample_key = list(data.keys())[0]
        sample_data = data[sample_key]
        
        print(f"\n📝 샘플 데이터 구조 (키: {sample_key}):")
        print("=" * 50)
        
        # 메타데이터 구조
        print("🔹 메타데이터 구조:")
        info_structure = {
            "stat_code": sample_data['info']['stat_code'],
            "item_code1": sample_data['info']['item_code1'], 
            "name": sample_data['info']['name'],
            "period": sample_data['info']['period'],
            "unit": sample_data['info']['unit']
        }
        print(json.dumps(info_structure, ensure_ascii=False, indent=2))
        
        # 데이터 구조 (처음 1개만)
        if sample_data['data']:
            print("\n🔹 데이터 항목 구조:")
            sample_row = sample_data['data'][0]
            print(json.dumps(sample_row, ensure_ascii=False, indent=2))
            
            print(f"\n🔹 데이터 컬럼 설명:")
            for col, value in sample_row.items():
                print(f"  - {col}: {type(value).__name__} (예: {value})")
        
        # 전체 데이터 요약
        print(f"\n📊 데이터 요약:")
        print("-" * 60)
        
        total_records = 0
        for key, info in data.items():
            stat_info = info['info']
            data_count = len(info['data'])
            total_records += data_count
            
            if data_count > 0:
                dates = [item['TIME'] for item in info['data']]
                first_date = min(dates)
                last_date = max(dates)
                status = "✅"
            else:
                first_date = last_date = "-"
                status = "❌"
                
            print(f"• {stat_info['name'][:40]:40} | {data_count:5}건 | {first_date} ~ {last_date} | {status}")
        
        print("-" * 60)
        print(f"📊 총 레코드 수: {total_records:,}건")
        
        # JSON 파일 정보
        if self.data_file.exists():
            file_size = self.data_file.stat().st_size / (1024*1024)  # MB
            file_time = datetime.fromtimestamp(self.data_file.stat().st_mtime)
            print(f"💾 파일 크기: {file_size:.2f}MB")
            print(f"🕒 마지막 수정: {file_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("=" * 80)
    
    def run_auto_update(self):
        """자동 업데이트 실행"""
        print("🚀 한국은행 ECOS 자동 데이터 업데이트")
        print("=" * 60)
        
        # 통계 목록 로드
        statistics = self.load_statistics_list()
        if not statistics:
            print("❌ 통계 목록을 로드할 수 없습니다.")
            return
        
        # 기존 데이터 확인
        existing_data = self.load_existing_data()
        is_initial_download = len(existing_data) == 0
        
        if is_initial_download:
            print("🆕 ecos_data.json 없음 → 전체 데이터 다운로드")
            update_only = False
        else:
            print("🔄 ecos_data.json 있음 → 증분 업데이트 (최신 날짜만)")
            update_only = True
        
        total_stats = len(statistics)
        success_count = 0
        error_count = 0
        update_count = 0
        
        today = datetime.now().strftime('%Y%m%d')
        
        for idx, stat_info in enumerate(statistics, 1):
            stat_code = stat_info['stat_code']
            item_code1 = stat_info['item_code1']
            name = stat_info['name']
            period = stat_info.get('period', 'D')  # 주기 정보 추출
            
            print(f"\n📊 [{idx}/{total_stats}] {name}")
            print(f"    📋 {stat_code} - {item_code1}")
            
            # 키 생성
            key = f"{stat_code}_{item_code1}"
            
            # 날짜 범위 결정
            if update_only:
                latest_date = self.get_latest_date(existing_data, stat_code, item_code1)
                if latest_date:
                    # 영업일 차이 계산 (1영업일 이내면 스킵)
                    try:
                        latest_dt = datetime.strptime(latest_date, '%Y%m%d')
                        today_dt = datetime.strptime(today, '%Y%m%d')
                        date_diff = (today_dt - latest_dt).days
                        
                        # 최신 데이터가 오늘과 같거나 이후면 스킵
                        if latest_date >= today:
                            print(f"    ✅ 이미 최신 데이터 (최종: {latest_date})")
                            if key not in existing_data:
                                existing_data[key] = {
                                    'info': stat_info,
                                    'data': [],
                                    'last_updated': datetime.now().isoformat()
                                }
                            success_count += 1
                            continue
                        # 1-3일 차이면 영업일 고려하여 스킵 (주말, 공휴일 고려)
                        elif date_diff <= 3 and date_diff >= 1:
                            print(f"    ✅ 영업일 기준 최신 (최종: {latest_date}, 차이: {date_diff}일)")
                            if key not in existing_data:
                                existing_data[key] = {
                                    'info': stat_info,
                                    'data': [],
                                    'last_updated': datetime.now().isoformat()
                                }
                            success_count += 1
                            continue
                    except:
                        # 날짜 파싱 실패 시 업데이트 진행
                        pass
                    
                    # 다음 날부터 오늘까지 업데이트
                    next_date = (datetime.strptime(latest_date, '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')
                    start_date, end_date = next_date, today
                    print(f"    🔄 증분 업데이트: {start_date} ~ {end_date}")
                else:
                    # 기존 데이터 없으면 전체 수집
                    start_date, end_date = self.get_date_range(stat_code, period)
                    print(f"    🆕 전체 수집: {start_date} ~ {end_date}")
            else:
                # 전체 수집
                start_date, end_date = self.get_date_range(stat_code, period)
                print(f"    🔄 전체 수집: {start_date} ~ {end_date}")
            
            # API 호출
            try:
                cycle = stat_info.get('period', 'D')  # 기본값: 일간
                new_data = self._make_api_request(stat_code, item_code1, start_date, end_date, cycle)
                
                if new_data:
                    # 데이터 저장
                    if key not in existing_data:
                        existing_data[key] = {
                            'info': stat_info,
                            'data': [],
                            'last_updated': datetime.now().isoformat()
                        }
                    
                    if update_only and key in existing_data and existing_data[key]['data']:
                        # 기존 데이터와 병합 (중복 제거)
                        existing_dates = {item['TIME'] for item in existing_data[key]['data']}
                        new_items = [item for item in new_data if item['TIME'] not in existing_dates]
                        existing_data[key]['data'].extend(new_items)
                        print(f"    ➕ 신규 데이터 추가: {len(new_items)}건")
                        if len(new_items) > 0:
                            update_count += 1
                    else:
                        # 전체 교체
                        existing_data[key]['data'] = new_data
                        print(f"    💾 데이터 저장: {len(new_data)}건")
                        update_count += 1
                    
                    # 날짜순 정렬
                    existing_data[key]['data'].sort(key=lambda x: x['TIME'])
                    existing_data[key]['last_updated'] = datetime.now().isoformat()
                    
                    success_count += 1
                    
                else:
                    print(f"    ❌ 데이터 수집 실패")
                    error_count += 1
                    
            except Exception as e:
                print(f"    ❌ 예외 발생: {e}")
                error_count += 1
        
        # 결과 저장
        if existing_data:
            self.save_data(existing_data)
        
        print("\n" + "=" * 60)
        print("📈 데이터 수집 완료")
        print(f"✅ 성공: {success_count}개")
        print(f"🔄 업데이트: {update_count}개")
        print(f"❌ 실패: {error_count}개")
        print(f"📊 총 통계: {len(existing_data)}개")
        
        # 데이터 구조 출력
        self.print_data_structure(existing_data)

def main():
    """메인 실행 함수"""
    # API 키 설정
    API_KEY = "AN3AJKNRJDS04779G6XP"  # 실제 API 키
    
    # 데이터 매니저 초기화
    manager = ECOSDataManager(API_KEY)
    
    print("🏦 한국은행 ECOS 자동 데이터 수집기")
    print(f"📅 실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 자동 업데이트 실행
    manager.run_auto_update()
    
    print(f"\n🏁 프로그램 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()