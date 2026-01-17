# db.py
# MongoDB 연결 관리 - 싱글톤 패턴

from pymongo import MongoClient
import config

class MongoDBManager:
    """MongoDB 연결을 싱글톤으로 관리하는 클래스"""
    _instance = None
    _client = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @property
    def client(self):
        if self._client is None:
            self._client = MongoClient(config.MONGO_URI)
        return self._client
    
    # ETF 데이터베이스
    @property
    def etf_db(self):
        return self.client[config.ETF_DATABASE]
    
    @property
    def ecos_db(self):
        return self.client[config.ECOS_DATABASE]
    
    # ETF 컬렉션
    @property
    def etf_master(self):
        return self.etf_db[config.COLLECTION_ETF_MASTER]
    
    @property
    def asset_pairs(self):
        return self.etf_db[config.COLLECTION_ASSET_PAIRS]
    
    @property
    def fund_prices(self):
        return self.etf_db[config.COLLECTION_FUND_PRICES]
    
    @property
    def synthetic_indices(self):
        return self.etf_db[config.COLLECTION_SYNTHETIC_INDICES]
    
    @property
    def market_summary(self):
        return self.etf_db[config.COLLECTION_MARKET_SUMMARY]
    
    # ECOS 컬렉션
    @property
    def ecos_prices(self):
        return self.ecos_db[config.COLLECTION_ECOS_PRICES]
    
    # 해외 배당 ETF 컬렉션 (dividen_model)
    @property
    def dividend_etf_summary(self):
        return self.etf_db[config.COLLECTION_DIVIDEND_ETF_SUMMARY]
    
    @property
    def dividend_etf_prices(self):
        return self.etf_db[config.COLLECTION_DIVIDEND_ETF_PRICES]
    
    @property
    def dividend_portfolio(self):
        return self.etf_db[config.COLLECTION_DIVIDEND_PORTFOLIO]


# 전역 인스턴스 (싱글톤)
db_manager = MongoDBManager()
