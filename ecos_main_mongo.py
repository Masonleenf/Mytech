import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from pymongo import MongoClient, ASCENDING

# ============= MongoDB 설정 =============
MONGO_URI = "mongodb+srv://rator9521_db_user:qwe343434@cluster0.d126rkt.mongodb.net/"
DATABASE_NAME = "ecos_database"
COLLECTION_ECOS_PRICES = "ecos_prices"

# 기존 로컬 파일 경로 (list.csv 읽기용)
LIST_CSV_FILE = "data/list.csv"

# MongoDB 연결
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]

class ECOSDataManager:
    def __init__(self, api_key):
        """
        한국은행 ECOS 데이터 관리 클래스 (MongoDB 버전)
        
        Args:
            api_key (str): 한국은행 API 키
        """
        self.api_key = api_key
        self.base_url = "https://ecos.bok.or.kr/api"
        
        # MongoDB 컬렉션
        self.collection = db[COLLECTION_ECOS_PRICES]
        
        # 인덱스 생성 (item_code1로 빠른 검색)
        self.collection.create_index([("item_code1", ASCENDING)], unique=True)
        
        # API 호출 제한
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
        """API 요청 실행"""
        cycle_mapping = {'D': 'D', 'M': 'M', 'Q': 'QQ', 'A': 'YY'}
        api_cycle = cycle_mapping.get(cycle.upper(), 'D')
        
        # 통계표별 항목코드 포맷팅
        if stat_code == '817Y002':
            formatted_item_code = f"0{str(item_code1).zfill(8)}"
        elif stat_code == '802Y001':
            formatted_item_code = f"0{str(item_code1).zfill(6)}"
        elif stat_code == '731Y001':
            formatted_item_code = f"0{str(item_code1).zfill(6)}"
        else:
            formatted_item_code = f"0{str(item_code1).zfill(6)}"
        
        url = f"{self.base_url}/StatisticSearch/{self.api_key}/json/kr/1/10000/{stat_code}/{api_cycle}/{start_date}/{end_date}/{formatted_item_code}"
        
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                
                print(f"API 요청: {stat_code} - {formatted_item_code} ({start_date}~{end_date})")
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                
                data = response.json()
                
                # 에러 체크
                if 'RESULT' in data:
                    if data['RESULT']['CODE'] != '0':
                        print(f"API 에러: {data['RESULT']['MESSAGE']}")
                        if data['RESULT']['CODE'] == '200':
                            return None
                        elif data['RESULT']['CODE'] == '602':
                            print("API 호출 제한, 대기 중...")
                            time.sleep(5)
                            continue
                        else:
                            return None
                
                # 데이터 확인
                if 'StatisticSearch' in data and 'row' in data['StatisticSearch']:
                    rows = data['StatisticSearch']['row']
                    print(f"✅ 데이터 수집: {len(rows)}건")
                    return rows
                else:
                    print("❌ 데이터 없음")
                    return None
                    
            except requests.exceptions.RequestException as e:
                print(f"요청 실패 ({attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    
            except Exception as e:
                print(f"예외 발생: {e}")
                return None
                
        return None
    
    def load_statistics_list(self):
        """list.csv에서 통계 목록 로드 (일별 데이터만)"""
        try:
            # CSV 읽기 (한글 인코딩 처리)
            try:
                df = pd.read_csv(LIST_CSV_FILE, encoding='utf-8-sig')
            except UnicodeDecodeError:
                df = pd.read_csv(LIST_CSV_FILE, encoding='cp949')
            
            df.columns = df.columns.str.strip()
            
            statistics = []
            for _, row in df.iterrows():
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
            
            print(f"📋 일별 통계 목록 로드: {len(statistics)}개")
            return statistics
            
        except Exception as e:
            print(f"❌ list.csv 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_latest_date_from_mongodb(self, item_code1):
        """MongoDB에서 최신 날짜 추출"""
        try:
            doc = self.collection.find_one({'item_code1': item_code1})
            if not doc or 'prices' not in doc or not doc['prices']:
                return None
            
            # 날짜 리스트에서 최신 날짜 찾기
            dates = [pd.to_datetime(p['Date'], format='%Y%m%d') for p in doc['prices']]
            latest_date = max(dates)
            return latest_date.strftime('%Y%m%d')
            
        except Exception as e:
            print(f"MongoDB 날짜 확인 오류: {e}")
            return None
    
    def _get_yesterday_value(self, item_code1, current_date):
        """어제 값 조회 (None 값 처리용)"""
        try:
            # 현재 날짜에서 1일 빼기
            current_dt = datetime.strptime(current_date, '%Y%m%d')
            yesterday_dt = current_dt - timedelta(days=1)
            yesterday_str = yesterday_dt.strftime('%Y%m%d')
            
            # MongoDB에서 해당 item_code1의 데이터 조회
            doc = self.collection.find_one({'item_code1': item_code1})
            if not doc or 'prices' not in doc:
                return None
            
            # 어제 날짜의 값 찾기
            for price_data in doc['prices']:
                if price_data['Date'] == yesterday_str:
                    return price_data['Close']
            
            # 어제 값이 없으면 최근 값 찾기 (최대 7일 전까지)
            for days_back in range(2, 8):
                check_date = current_dt - timedelta(days=days_back)
                check_date_str = check_date.strftime('%Y%m%d')
                
                for price_data in doc['prices']:
                    if price_data['Date'] == check_date_str:
                        return price_data['Close']
            
            return None
            
        except Exception as e:
            print(f"어제 값 조회 오류: {e}")
            return None
    
    def save_data_to_mongodb(self, item_code1, stat_info, api_data, is_update=False):
        """API 데이터를 MongoDB에 저장"""
        try:
            # API 데이터를 정제
            new_data = []
            for row in api_data:
                data_value = row['DATA_VALUE']
                
                # None 값 처리: 어제 값 사용
                if data_value is None or data_value == '':
                    yesterday_value = self._get_yesterday_value(item_code1, row['TIME'])
                    if yesterday_value is not None:
                        data_value = yesterday_value
                        print(f"    ⚠️  None 값 감지, 어제 값 사용: {row['TIME']} -> {yesterday_value}")
                    else:
                        print(f"    ❌ None 값이고 어제 값도 없음: {row['TIME']}")
                        continue  # 이 데이터는 건너뛰기
                
                try:
                    close_value = float(data_value)
                except (ValueError, TypeError) as e:
                    print(f"    ❌ 데이터 변환 실패: {data_value} -> {e}")
                    continue
                
                new_data.append({
                    'Date': row['TIME'],
                    'Close': close_value
                })
            
            # 날짜순 정렬
            new_data = sorted(new_data, key=lambda x: x['Date'])
            
            if is_update:
                # 기존 문서에 추가
                existing_doc = self.collection.find_one({'item_code1': item_code1})
                
                if existing_doc and 'prices' in existing_doc:
                    # 중복 제거를 위해 기존 데이터와 병합
                    existing_dates = {p['Date'] for p in existing_doc['prices']}
                    new_unique_data = [d for d in new_data if d['Date'] not in existing_dates]
                    
                    if new_unique_data:
                        self.collection.update_one(
                            {'item_code1': item_code1},
                            {'$push': {'prices': {'$each': new_unique_data}}}
                        )
                        print(f"    ➕ 데이터 업데이트: {len(new_unique_data)}건 추가")
                    else:
                        print(f"    ✅ 새 데이터 없음 (중복)")
                else:
                    # 문서는 있지만 prices가 없는 경우
                    self.collection.update_one(
                        {'item_code1': item_code1},
                        {'$set': {'prices': new_data}}
                    )
                    print(f"    💾 데이터 저장: {len(new_data)}건")
            else:
                # 새 문서 생성 (upsert)
                self.collection.update_one(
                    {'item_code1': item_code1},
                    {'$set': {
                        'item_code1': item_code1,
                        'stat_code': stat_info['stat_code'],
                        'name': stat_info['name'],
                        'period': stat_info['period'],
                        'unit': stat_info['unit'],
                        'prices': new_data
                    }},
                    upsert=True
                )
                print(f"    💾 새 문서 저장: {len(new_data)}건")
            
        except Exception as e:
            print(f"    ❌ MongoDB 저장 실패: {e}")
    
    def run_auto_update(self):
        """자동 업데이트 실행 (MongoDB 버전)"""
        print("🚀 한국은행 ECOS 자동 데이터 업데이트 (MongoDB)")
        print("=" * 60)
        
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
            
            start_date = "20000101"
            end_date = today
            is_update = False
            
            # MongoDB에서 최신 날짜 확인
            latest_date = self.get_latest_date_from_mongodb(item_code1)
            
            if latest_date:
                try:
                    latest_dt = datetime.strptime(latest_date, '%Y%m%d')
                    today_dt = datetime.strptime(today, '%Y%m%d')
                    
                    if latest_date >= today:
                        print(f"    ✅ 이미 최신 (최종: {latest_date})")
                        success_count += 1
                        continue
                    
                    # 다음 날부터 업데이트
                    next_date = (latest_dt + timedelta(days=1)).strftime('%Y%m%d')
                    start_date = next_date
                    is_update = True
                    print(f"    🔄 증분 업데이트: {start_date} ~ {end_date}")
                    
                except:
                    print(f"    🆕 전체 수집: {start_date} ~ {end_date}")
            else:
                print(f"    🆕 새 데이터 생성: {start_date} ~ {end_date}")
            
            # API 호출
            try:
                api_data = self._make_api_request(stat_code, item_code1, start_date, end_date, 'D')
                
                if api_data:
                    self.save_data_to_mongodb(item_code1, stat_info, api_data, is_update)
                    success_count += 1
                    
                    if is_update:
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
        print(f"📁 저장 위치: MongoDB - {DATABASE_NAME}.{COLLECTION_ECOS_PRICES}")
        
        # 저장된 문서 수 확인
        doc_count = self.collection.count_documents({})
        print(f"📊 총 문서 수: {doc_count}개")

def main():
    """메인 실행 함수"""
    # API 키 설정
    API_KEY = "AN3AJKNRJDS04779G6XP"
    
    print("=" * 60)
    print("한국은행 ECOS 데이터 관리 시스템 (MongoDB 버전)")
    print("=" * 60)
    print(f"📅 실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 데이터 매니저 초기화
    manager = ECOSDataManager(API_KEY)
    
    # 자동 업데이트 실행
    manager.run_auto_update()
    
    print("\n" + "=" * 60)
    print("모든 작업 완료!")
    print("=" * 60)

if __name__ == "__main__":
    main()