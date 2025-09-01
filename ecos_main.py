import requests
import pandas as pd
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

class ECOSDataManager:
    def __init__(self, api_key, data_dir="./data"):
        """
        한국은행 ECOS 데이터 관리 클래스 (개별 CSV 파일 저장 방식)
        
        Args:
            api_key (str): 한국은행 API 키
            data_dir (str): 데이터 저장 디렉토리
        """
        self.api_key = api_key
        self.base_url = "https://ecos.bok.or.kr/api"
        self.data_dir = Path(data_dir)
        
        # CSV 파일 저장 경로 (data_manager.py와 유사한 구조)
        self.price_data_dir = self.data_dir / "ecos_prices"
        self.list_file = self.data_dir / "list.csv"
        
        # 폴더 생성
        self.price_data_dir.mkdir(parents=True, exist_ok=True)
        
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
            list: API 응답 데이터 리스트 또는 None
        """
        # 주기 형식 변환
        cycle_mapping = {'D': 'D', 'M': 'M', 'Q': 'QQ', 'A': 'YY'}
        api_cycle = cycle_mapping.get(cycle.upper(), 'D')
        
        # 통계표별 항목코드 포맷팅 규칙
        if stat_code == '817Y002':  # 금리
            formatted_item_code = f"0{str(item_code1).zfill(8)}"
        elif stat_code == '802Y001':  # 주가지수
            formatted_item_code = f"0{str(item_code1).zfill(6)}"
        elif stat_code == '731Y001':  # 환율
            formatted_item_code = f"0{str(item_code1).zfill(6)}"
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
        list.csv에서 통계 목록 로드 (일별 데이터만)
        
        Returns:
            list: 통계 정보 리스트
        """
        try:
            df = pd.read_csv(self.list_file)
            
            # 컬럼명 정리
            df.columns = df.columns.str.strip()
            
            statistics = []
            for _, row in df.iterrows():
                # 일별 데이터만 처리 (period가 'D'인 것만)
                if str(row['period']).strip().upper() != 'D':
                    continue
                    
                stat_info = {
                    'stat_code': str(row['stat_code']).strip(),
                    'item_code1': int(row['item_code1']),
                    'name': str(row['name']).strip(),
                    'period': str(row['period']).strip(),
                    'unit': str(row['단위']).strip()
                }
                statistics.append(stat_info)
            
            print(f"📋 일별 통계 목록 로드 완료: {len(statistics)}개")
            return statistics
            
        except Exception as e:
            print(f"❌ list.csv 로드 실패: {e}")
            return []
    
    def get_csv_file_path(self, item_code1):
        """CSV 파일 경로 생성"""
        return self.price_data_dir / f"{item_code1}.csv"
    
    def get_latest_date_from_csv(self, csv_file_path):
        """
        CSV 파일에서 최신 날짜 추출
        
        Args:
            csv_file_path (Path): CSV 파일 경로
            
        Returns:
            str: 최신 날짜 (YYYYMMDD) 또는 None
        """
        try:
            if not csv_file_path.exists():
                return None
                
            df = pd.read_csv(csv_file_path)
            if df.empty or 'Date' not in df.columns:
                return None
                
            # 날짜 컬럼을 정렬해서 최신 날짜 찾기
            df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d', errors='coerce')
            df = df.dropna(subset=['Date'])
            
            if df.empty:
                return None
                
            latest_date = df['Date'].max()
            return latest_date.strftime('%Y%m%d')
            
        except Exception as e:
            print(f"CSV 파일 날짜 확인 오류: {e}")
            return None
    
    def save_data_to_csv(self, item_code1, stat_info, api_data, is_update=False):
        """
        API 데이터를 CSV 파일로 저장
        
        Args:
            item_code1 (int): 항목코드1
            stat_info (dict): 통계 정보
            api_data (list): API에서 받은 데이터
            is_update (bool): 업데이트 모드인지 여부
        """
        try:
            csv_file_path = self.get_csv_file_path(item_code1)
            
            # API 데이터를 DataFrame으로 변환
            new_df = pd.DataFrame(api_data)
            
            # 필요한 컬럼만 추출: 날짜, 값
            new_df = new_df[['TIME', 'DATA_VALUE']].copy()
            new_df.columns = ['Date', 'Close']
            
            # 데이터 타입 변환
            new_df['Date'] = pd.to_datetime(new_df['Date'], format='%Y%m%d', errors='coerce')
            new_df['Close'] = pd.to_numeric(new_df['Close'], errors='coerce')
            
            # 결측치 제거
            new_df = new_df.dropna()
            
            # 날짜순 정렬
            new_df = new_df.sort_values('Date')
            
            # 날짜를 다시 YYYYMMDD 문자열로 변환 (CSV 저장용)
            new_df['Date'] = new_df['Date'].dt.strftime('%Y%m%d')
            
            if is_update and csv_file_path.exists():
                # 기존 파일이 있으면 읽어서 병합
                try:
                    existing_df = pd.read_csv(csv_file_path)
                    
                    # 중복 제거하면서 병합
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    combined_df = combined_df.drop_duplicates(subset=['Date'], keep='last')
                    
                    # 날짜순 정렬
                    combined_df['Date'] = pd.to_datetime(combined_df['Date'], format='%Y%m%d')
                    combined_df = combined_df.sort_values('Date')
                    combined_df['Date'] = combined_df['Date'].dt.strftime('%Y%m%d')
                    
                    # 저장
                    combined_df.to_csv(csv_file_path, index=False)
                    print(f"    ➕ 기존 데이터 업데이트: {len(new_df)}건 추가")
                    
                except Exception as e:
                    print(f"    ❌ 기존 파일 병합 실패: {e}")
                    # 실패하면 새 파일로 저장
                    new_df.to_csv(csv_file_path, index=False)
            else:
                # 새 파일로 저장
                new_df.to_csv(csv_file_path, index=False)
                print(f"    💾 새 파일 저장: {len(new_df)}건")
            
        except Exception as e:
            print(f"    ❌ CSV 저장 실패: {e}")
    
    def run_auto_update(self):
        """자동 업데이트 실행 (개별 CSV 파일 방식)"""
        print("🚀 한국은행 ECOS 자동 데이터 업데이트 (CSV 파일)")
        print("=" * 60)
        
        # 통계 목록 로드 (일별만)
        statistics = self.load_statistics_list()
        if not statistics:
            print("❌ 통계 목록을 로드할 수 없습니다.")
            return
        
        total_stats = len(statistics)
        success_count = 0
        error_count = 0
        update_count = 0
        
        today = datetime.now().strftime('%Y%m%d')
        
        for idx, stat_info in enumerate(statistics, 1):
            stat_code = stat_info['stat_code']
            item_code1 = stat_info['item_code1']
            name = stat_info['name']
            
            print(f"\n📊 [{idx}/{total_stats}] {name}")
            print(f"    📋 {stat_code} - {item_code1}")
            
            csv_file_path = self.get_csv_file_path(item_code1)
            
            # 시작 날짜 결정
            start_date = "20000101"  # 요청사항: 2024-01-01부터
            end_date = today
            is_update = False
            
            # 기존 파일이 있으면 최신 날짜부터 업데이트
            if csv_file_path.exists():
                latest_date = self.get_latest_date_from_csv(csv_file_path)
                if latest_date:
                    try:
                        latest_dt = datetime.strptime(latest_date, '%Y%m%d')
                        today_dt = datetime.strptime(today, '%Y%m%d')
                        date_diff = (today_dt - latest_dt).days
                        
                        # 최신 데이터가 오늘과 같거나 이후면 스킵
                        if latest_date >= today:
                            print(f"    ✅ 이미 최신 데이터 (최종: {latest_date})")
                            success_count += 1
                            continue
                        # 1-3일 차이면 영업일 고려하여 스킵 (주말, 공휴일 고려)
                        elif date_diff <= 3 and date_diff >= 1:
                            print(f"    ✅ 영업일 기준 최신 (최종: {latest_date}, 차이: {date_diff}일)")
                            success_count += 1
                            continue
                    except:
                        # 날짜 파싱 실패 시 업데이트 진행
                        pass
                    
                    # 다음 날부터 오늘까지 업데이트
                    next_date = (datetime.strptime(latest_date, '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')
                    start_date = next_date
                    is_update = True
                    print(f"    🔄 증분 업데이트: {start_date} ~ {end_date}")
                else:
                    print(f"    🆕 전체 수집: {start_date} ~ {end_date}")
            else:
                print(f"    🆕 새 파일 생성: {start_date} ~ {end_date}")
            
            # API 호출
            try:
                api_data = self._make_api_request(stat_code, item_code1, start_date, end_date, 'D')
                
                if api_data:
                    # CSV 파일로 저장
                    self.save_data_to_csv(item_code1, stat_info, api_data, is_update)
                    success_count += 1
                    
                    if is_update or not csv_file_path.exists():
                        update_count += 1
                else:
                    print(f"    ❌ 데이터 수집 실패")
                    error_count += 1
                    
            except Exception as e:
                print(f"    ❌ 예외 발생: {e}")
                error_count += 1
        
        print("\n" + "=" * 60)
        print("📈 데이터 수집 완료")
        print(f"✅ 성공: {success_count}개")
        print(f"🔄 업데이트: {update_count}개")
        print(f"❌ 실패: {error_count}개")
        print(f"📁 저장 경로: {self.price_data_dir}")
        
        # 생성된 파일 목록 출력
        csv_files = list(self.price_data_dir.glob("*.csv"))
        print(f"📊 총 CSV 파일 수: {len(csv_files)}개")

def main():
    """메인 실행 함수"""
    # API 키 설정
    API_KEY = "AN3AJKNRJDS04779G6XP"  # 실제 API 키
    
    # 데이터 매니저 초기화
    manager = ECOSDataManager(API_KEY)
    
    print("🏦 한국은행 ECOS 자동 데이터 수집기 (CSV 버전)")
    print(f"📅 실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 자동 업데이트 실행
    manager.run_auto_update()
    
    print(f"\n🏁 프로그램 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()