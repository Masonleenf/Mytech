# dividend_data_loader.py
# etf_summary.json 데이터를 MongoDB에 적재하는 스크립트

import json
import os
from db import db_manager

def load_etf_summary_to_mongodb():
    """dividend_data/etf_summary.json을 MongoDB에 적재"""
    data_path = os.path.join(os.path.dirname(__file__), 'dividend_data', 'etf_summary.json')
    
    if not os.path.exists(data_path):
        print(f"❌ 데이터 파일 없음: {data_path}")
        return False
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            etf_list = json.load(f)
        
        print(f"📊 {len(etf_list)}개 ETF 데이터 로드됨")
        
        # 기존 데이터 삭제
        db_manager.dividend_etf_summary.delete_many({})
        print("  🗑️ 기존 데이터 삭제")
        
        # 새 데이터 삽입
        if etf_list:
            result = db_manager.dividend_etf_summary.insert_many(etf_list)
            print(f"  ✅ {len(result.inserted_ids)}개 문서 삽입 완료")
        
        return True
        
    except Exception as e:
        print(f"❌ MongoDB 적재 실패: {e}")
        return False


def verify_data():
    """적재된 데이터 확인"""
    count = db_manager.dividend_etf_summary.count_documents({})
    print(f"\n📋 dividend_etf_summary 컬렉션: {count}개 문서")
    
    if count > 0:
        sample = db_manager.dividend_etf_summary.find_one({}, {'_id': 0})
        print(f"  샘플 데이터: {sample.get('ticker', 'N/A')} - {sample.get('name', 'N/A')}")


if __name__ == '__main__':
    print("=" * 50)
    print("배당 ETF 데이터 MongoDB 적재")
    print("=" * 50)
    
    load_etf_summary_to_mongodb()
    verify_data()
    
    print("\n완료!")
