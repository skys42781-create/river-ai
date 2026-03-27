import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# [1. 기본 설정 및 한글 폰트]
st.set_page_config(page_title="한강 유량 예측 시스템", layout="wide")
# 스트림릿 클라우드(리눅스) 환경에서는 나눔고딕을 기본으로 설정하는 것이 좋습니다.
# 만약 폰트 에러가 나면 이 부분은 생략해도 수치 표시는 잘 됩니다.

st.title("🌊 한강 실시간 유량 예측 시스템 (GitHub Relay)")
st.markdown("---")

# [2. 인공지능 모델 불러오기]
@st.cache_resource
def get_river_model():
    # compile=False 옵션으로 버전 차이로 인한 에러를 원천 차단합니다.
    return load_model('river_model_final.h5', compile=False)

model = get_river_model()

# [3. 깃허브에 올린 최신 CSV 데이터 로드]
def load_github_data():
    # 근우님의 깃허브 Raw 데이터 주소
    csv_url = "https://raw.githubusercontent.com/skys42781-create/river-ai/main/latest_river_data.csv"
    
    try:
        # 깃허브 파일은 속도가 매우 빠르고 접속 제한이 없습니다.
        df = pd.read_csv(csv_url)
        return df
    except Exception as e:
        st.error(f"⚠️ 깃허브에서 데이터를 가져오지 못했습니다: {e}")
        return None

# [4. 메인 대시보드 구성]
st.sidebar.header("🕹️ 시스템 제어")
if st.sidebar.button("🔄 최신 분석 결과 업데이트"):
    st.rerun()

# 데이터 로드 실행
df_raw = load_github_data()

if df_raw is not None:
    # --- 데이터 무결성 검토 및 전처리 ---
    # CSV에서 읽어온 컬럼명을 표준화합니다.
    df = df_raw.copy()
    
    # 숫자형 변환 (혹시 모를 문자열 방지)
    df['강수량(mm)'] = pd.to_numeric(df['강수량(mm)'], errors='coerce')
    df['유량(m3/s)'] = pd.to_numeric(df['유량(m3/s)'], errors='coerce')
    
    # 결측치 제거 및 물리적 오류(음수) 필터링
    df = df.dropna().query("`강수량(mm)` >= 0 and `유량(m3/s)` >= 0")
    df = df.sort_values('년월일(yyyyMMdd)')
    
    if len(df) >= 7:
        st.success(f"✅ 데이터 동기화 완료: {df['년월일(yyyyMMdd)'].iloc[-1]} 기준")
        
        # [5. 인공지능 예측 실행]
        # 모델 학습 시 사용했던 MinMaxScaler와 동일한 구조로 스케일링
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[['강수량(mm)', '유량(m3/s)']].values)
        
        # 최근 7일 데이터를 입력값으로 사용 (1, 7, 2 구조)
        recent_input = scaled_data[-7:].reshape(1, 7, 2)
        pred_scaled = model.predict(recent_input, verbose=0)[0]
        
        # 역정규화 (예측된 스케일 값을 실제 m3/s 단위로 복원)
        def inverse_val(s_val):
            dummy = np.zeros((1, 2))
            dummy[0, 1] = s_val
            return scaler.inverse_transform(dummy)[0, 1]
        
        # [6. 결과 화면 출력]
        st.subheader("📅 향후 3일간 유량 예측 결과")
        c1, c2, c3 = st.columns(3)
        
        labels = ["내일 (T+1)", "모레 (T+2)", "글피 (T+3)"]
        preds = [inverse_val(p) for p in pred_scaled]
        
        for i, (label, val) in enumerate(zip(labels, preds)):
            with [c1, c2, c3][i]:
                st.metric(label, f"{val:.2f} m³/s")
        
        # [7. 관측 데이터 시각화]
        st.divider()
        st.subheader("📊 최근 10일간의 수문 데이터 흐름")
        st.dataframe(df.tail(10), use_container_width=True)
        
        # 간단한 라인 차트 추가
        st.line_chart(df.tail(15).set_index('년월일(yyyyMMdd)')['유량(m3/s)'])

    else:
        st.warning("예측을 위한 데이터(최소 7일치)가 부족합니다. CSV 파일을 확인해 주세요.")
else:
    st.info("데이터를 불러오는 중입니다. 잠시만 기다려 주세요.")
