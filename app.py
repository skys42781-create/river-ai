import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import concurrent.futures # 병렬 처리를 위한 도구

st.set_page_config(page_title="한강 유량 초고속 예측", layout="wide")
st.title("⚡ 실시간 한강 유량 예측 시스템 (최적화 버전)")

# [1. 모델 캐싱] 앱 실행 시 딱 한 번만 불러옵니다.
@st.cache_resource
def get_model():
    return load_model('river_model_final.h5', compile=False)

model = get_model()

# [2. API 호출 함수 (병렬 처리용)]
def fetch_url(url):
    return pd.read_xml(url, xpath=".//item", storage_options={'timeout': 10})

# [3. 데이터 수집 최적화] 1시간 동안은 다시 불러오지 않고 저장된 값을 씁니다.
@st.cache_data(ttl=3600)
def get_optimized_data():
    now = datetime.now()
    curr_time = now.strftime("%Y%m%d%H%M")
    start_time = (now - timedelta(days=14)).strftime("%Y%m%d%H%M") # 14일로 단축
    
    api_key = "5283B235-DEA8-4DC9-9E36-D7A8A9AEA79F"
    wl_url = f"https://api.hrfco.go.kr/{api_key}/waterlevel/list/1D/2004680/{start_time}/{curr_time}.xml"
    rf_url = f"https://api.hrfco.go.kr/{api_key}/rainfall/list/1D/20044080/{start_time}/{curr_time}.xml"
    
    # 병렬로 두 API 동시 호출 (속도 향상 핵심)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_wl = executor.submit(fetch_url, wl_url)
        future_rf = executor.submit(fetch_url, rf_url)
        wl_df = future_wl.result()[['ymd', 'fw']]
        rf_df = future_rf.result()[['ymd', 'rf']]
        
    df = pd.merge(rf_df, wl_df, on='ymd').rename(columns={'ymd':'날짜', 'rf':'강수량', 'fw':'유량'})
    
    # 무결성 검토
    df[['강수량', '유량']] = df[['강수량', '유량']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna().query("강수량 >= 0 and 유량 >= 0")
    return df.sort_values('날짜')

# [4. 메인 실행]
if st.button("🚀 최신 데이터 분석 시작"):
    with st.spinner("⚡ 데이터를 병렬로 수집하고 분석 중입니다..."):
        df = get_optimized_data()
        
        if len(df) >= 7:
            # 예측 준비
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(df[['강수량', '유량']].values)
            
            # 예측 및 수치 복원
            pred_scaled = model.predict(scaled[-7:].reshape(1, 7, 2), verbose=0)[0]
            def to_real(val):
                dummy = np.zeros((1, 2)); dummy[0, 1] = val
                return scaler.inverse_transform(dummy)[0, 1]
            
            # 수치 표시
            st.subheader("📅 향후 3일 예측 유량")
            c1, c2, c3 = st.columns(3)
            res = [to_real(p) for p in pred_scaled]
            c1.metric("내일 (T+1)", f"{res[0]:.2f} m³/s")
            c2.metric("모레 (T+2)", f"{res[1]:.2f} m³/s")
            c3.metric("글피 (T+3)", f"{res[2]:.2f} m³/s")

            # 오차율 확인용 최근 데이터
            st.divider()
            st.subheader("📊 최근 관측 데이터 (무결성 확인)")
            st.table(df.tail(5)) # DataFrame보다 가벼운 Table 사용
        else:
            st.error("데이터 수집량이 부족합니다. 잠시 후 다시 시도해 주세요.")
