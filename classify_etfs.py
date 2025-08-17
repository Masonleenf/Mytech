import json
import os
import re

# 원본 파일과 결과 파일 경로 설정
DATA_DIR = "data"
SOURCE_FILE = os.path.join(DATA_DIR, "etf_master.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "etf_master_classified.json")

def classify_etf(name):
    """ETF 이름을 분석하여 SAA와 TAA 클래스를 반환하는 함수"""
    name = name.upper() # 분석을 위해 이름을 대문자로 변환

    # 1. 파생/특수 상품 우선 분류 (SAA: 기타파생)
    if any(keyword in name for keyword in ["레버리지", "인버스", "인버스2X", "선물"]):
        if "국채" in name or "채권" in name:
            saa = "기타파생"
            taa = "채권파생"
        else:
            saa = "기타파생"
            taa = "주식파생"
        return saa, taa
    if "커버드콜" in name:
        return "기타파생", "커버드콜"
    
    # 2. 자산 유형 기반 분류
    if any(keyword in name for keyword in ["CD금리", "KOFR", "머니마켓", "단기자금", "단기채권"]):
        return "현금성자산", "단기채권/금리"
    if any(keyword in name for keyword in ["국고채", "국채", "회사채", "은행채", "종합채권", "단기통안채"]):
        if any(keyword in name for keyword in ["미국", "달러"]):
             return "해외채권", "미국채권"
        return "국내채권", "종합채권"
    if any(keyword in name for keyword in ["리츠", "부동산", "인프라"]):
        return "대체투자", "리츠/부동산"
    if any(keyword in name for keyword in ["금", "골드", "원유", "농산물", "구리", "원자재"]):
        return "대체투자", "원자재"
    if any(keyword in name for keyword in ["TDF", "TRF", "채권혼합", "주식혼합", "자산배분"]):
        return "혼합자산", "자산배분"

    # 3. 지역 기반 분류 (SAA: 해외주식/국내주식)
    is_foreign = True
    if any(keyword in name for keyword in ["미국", "S&P", "나스닥", "다우존스"]):
        saa = "해외주식"; taa = "미국"
    elif any(keyword in name for keyword in ["차이나", "중국", "항셍"]):
        saa = "해외주식"; taa = "중국"
    elif any(keyword in name for keyword in ["일본", "니케이"]):
        saa = "해외주식"; taa = "일본"
    elif any(keyword in name for keyword in ["유럽", "유로"]):
        saa = "해외주식"; taa = "유럽"
    elif "인도" in name:
        saa = "해외주식"; taa = "인도"
    elif "베트남" in name:
        saa = "해외주식"; taa = "베트남"
    elif "글로벌" in name or "MSCI" in name:
        saa = "해외주식"; taa = "글로벌"
    else:
        is_foreign = False
        saa = "국내주식"; taa = "국내일반"

    # 4. 테마/섹터 기반 TAA 세분화
    if any(keyword in name for keyword in ["반도체", "비메모리"]):
        taa += "반도체" if is_foreign else "반도체"
    elif "2차전지" in name or "이차전지" in name or "배터리" in name:
        taa += "2차전지" if is_foreign else "2차전지"
    elif "AI" in name or "인공지능" in name:
        taa += "AI" if is_foreign else "AI"
    elif "IT" in name or "테크" in name or "소프트웨어" in name:
        taa += "IT" if is_foreign else "IT"
    elif "바이오" in name or "헬스케어" in name:
        taa += "바이오" if is_foreign else "바이오"
    elif any(keyword in name for keyword in ["고배당", "배당"]):
        taa += "배당주" if is_foreign else "배당주"
    elif "친환경" in name or "클린에너지" in name or "수소" in name:
        taa += "친환경" if is_foreign else "친환경"
    elif "로봇" in name:
        taa = "로봇"
    elif "금융" in name:
        taa = "금융"
    elif "코스닥" in name:
        taa = "코스닥"
    elif "200" in name or "코스피" in name:
        taa = "국내대형주"

    # 기본 TAA값 정리
    taa = re.sub(r"^(미국|중국|일본|유럽|인도|베트남|글로벌|국내일반)", "", taa)
    if is_foreign:
        prefix = "글로벌"
        if "미국" in taa: prefix = "미국"
        elif "중국" in taa: prefix = "중국"
        #...
        taa = prefix + (taa if taa else "주식")
    
    return saa, taa.replace("국내일반", "") or "기타테마"


def main():
    """메인 실행 함수"""
    print(f"'{SOURCE_FILE}' 파일을 읽어 분류를 시작합니다.")
    try:
        with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
            etf_list = json.load(f)
    except FileNotFoundError:
        print(f"오류: 원본 파일인 '{SOURCE_FILE}'을 찾을 수 없습니다.")
        return
    except Exception as e:
        print(f"오류: 파일 로딩 중 문제가 발생했습니다 - {e}")
        return

    classified_list = []
    for etf in etf_list:
        try:
            saa, taa = classify_etf(etf['name'])
            etf['saa_class'] = saa
            etf['taa_class'] = taa
            classified_list.append(etf)
        except Exception as e:
            print(f"경고: '{etf.get('name', '알수없음')}' 분류 중 오류 발생 - {e}")
            # 오류가 발생해도 원본 데이터는 유지
            classified_list.append(etf)

    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(classified_list, f, ensure_ascii=False, indent=4)
        print("-" * 50)
        print(f"성공! 분류가 완료된 파일이 '{OUTPUT_FILE}' 경로에 저장되었습니다.")
        print("이제 이 파일의 이름을 'etf_master.json'으로 변경하여 사용하세요.")
        print("-" * 50)
    except Exception as e:
        print(f"오류: 결과 파일 저장 중 문제가 발생했습니다 - {e}")

if __name__ == '__main__':
    main()