import json
import pandas as pd

# 입력 파일과 출력 파일 경로 설정
source_file = 'data/etf_master.json'
output_file = 'App/mytechapp/assets/data/asset_pairs.json' # Flutter 프로젝트 assets 폴더로 옮길 최종 파일

def create_asset_pairs():
    """
    etf_master.json에서 SAA와 TAA 조합을 추출하고 중복을 제거하여
    asset_pairs.json 파일을 생성합니다.
    """
    print(f"'{source_file}' 파일을 읽어 자산 조합을 생성합니다...")
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            etf_list = json.load(f)
        
        df = pd.DataFrame(etf_list)
        
        # SAA와 TAA 클래스 컬럼명을 소문자로 통일
        df.rename(columns={'SAAclass': 'saa_class', 'TAAclass': 'taa_class'}, inplace=True)
        
        # 두 컬럼을 선택하고, 중복된 행을 제거
        asset_pairs_df = df[['saa_class', 'taa_class']].drop_duplicates()
        
        # 결측값(NaN)이나 "미분류" 항목은 제외
        asset_pairs_df = asset_pairs_df.dropna()
        asset_pairs_df = asset_pairs_df[asset_pairs_df['saa_class'] != '미분류']
        
        # JSON 형식으로 변환
        result = asset_pairs_df.to_dict('records')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
            
        print(f"성공! 총 {len(result)}개의 자산 조합을 '{output_file}'에 저장했습니다.")

    except FileNotFoundError:
        print(f"오류: '{source_file}'을 찾을 수 없습니다.")
    except Exception as e:
        print(f"오류: {e}")

if __name__ == '__main__':
    create_asset_pairs()