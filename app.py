import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# [1. 환경 설정]
st.set_page_config(page_title="한강 유량 실시간 예측", layout="wide")
plt.rc('font', family='NanumBarunGothic') 
plt.rcParams['axes.unicode_minus'] = False

st.title("🌊 한강홍수통제소 API 실시간 유량 예측 시스템")

# [2. 모델 로드]
@st.cache_resource
def get_model():
    return load_model('river_model_final.h5', compile=False)

model = get_model()

# [3. API 데이터 자동 수집 및 무결성 검토 함수]
def fetch_and_validate_data():
    curr_time = datetime.now().strftime("%Y%m%d%H%M")
    
    # API URL 설정 (사용자 제공 주소 기반)
    wl_url = f"https://api.hrfco.go.kr/5283B235-DEA8-4DC9-9E36-D7A8A9AEA79F/waterlevel/list/1D/2004680/200701010000/{curr_time}.xml"
    rf_url = f"https://api.hrfco.go.kr/5283B235-DEA8-4DC9-9E36-D7A8A9AEA79F/rainfall/list/1D/20044080/199701010000/{curr_time}.xml"
    
    try:
        with st.spinner("📡 한강홍수통제소에서 최신 데이터를 가져오는 중..."):
            # XML 데이터 읽기 및 컬럼명 표준화
            wl_df = pd.read_xml(wl_url, xpath=".//item")[['ymd', 'fw']] # fw: 유량
            rf_df = pd.read_xml(rf_url, xpath=".//item")[['ymd', 'rf']] # rf: 강우량
            
            # 데이터 병합
            df = pd.merge(rf_df, wl_df, on='ymd').rename(columns={
                'ymd': '년월일(yyyyMMdd)', 'rf': '강수량(mm)', 'fw': '유량(m3/s)'
            })
            
            # --- 무결성 검토 (Integrity Check) ---
            # 1. 수치형 변환 및 결측치 확인
            df['강수량(mm)'] = pd.to_numeric(df['강수량(mm)'], errors='coerce')
            df['유량(m3/s)'] = pd.to_numeric(df['유량(m3/s)'], errors='coerce')
            
            # 2. 음수 및 물리적 오류 제거
            df = df[(df['강수량(mm)'] >= 0) & (df['유량(m3/s)'] >= 0)]
            
            # 3. 결측치 보간 (선형 보간법)
            df = df.interpolate(method='linear').dropna()
            
            return df.sort_values('년월일(yyyyMMdd)')
    except Exception as e:
        st.error(f"❌ 데이터 수집 중 오류 발생: {e}")
        return None

# [4. 메인 화면 구성]
if st.button("🔄 실시간 데이터 업데이트 및 예측 실행"):
    df = fetch_and_validate_data()
    
    if df is not None and len(df) >= 7:
        st.success(f"✅ 데이터 무결성 검토 완료 ({df['년월일(yyyyMMdd)'].iloc[-1]} 기준)")
        
        # 데이터 전처리
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[['강수량(mm)', '유량(m3/s)']].values)
        
        # 예측 수행 (최근 7일 데이터 사용)
        recent_input = scaled_data[-7:].reshape(1, 7, 2)
        pred_scaled = model.predict(recent_input)[0]
        
        # 역정규화 함수
        def to_real(val):
            dummy = np.zeros((1, 2))
            dummy[0, 1] = val
            return scaler.inverse_transform(dummy)[0, 1]
        
        # 결과 표시
        st.subheader("📅 향후 3일간 유량 예측 수치")
        cols = st.columns(3)
        labels = ["내일(T+1)", "모레(T+2)", "글피(T+3)"]
        for i in range(3):
            cols[i].metric(labels[i], f"{to_real(pred_scaled[i]):.2f} m³/s")
            
        # 데이터 테이블 출력
        st.divider()
        st.subheader("📊 최근 관측 자료 (무결성 검토 완료)")
        st.dataframe(df.tail(10), use_container_width=True)
    else:
        st.warning("예측을 위한 충분한 데이터(최소 7일치)를 불러오지 못했습니다.")
else:
    st.info("위의 버튼을 눌러 실시간 API 데이터를 불러오세요.")
