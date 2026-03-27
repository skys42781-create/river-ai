import streamlit as st
import pandas as pd
import numpy as np
import requests # 안정적인 연결을 위해 추가
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import io

st.set_page_config(page_title="한강 유량 실시간 예측", layout="wide")
st.title("🌊 한강 유량 실시간 예측 시스템")

@st.cache_resource
def get_model():
    return load_model('river_model_final.h5', compile=False)

model = get_model()

# [안정성 강화] 데이터를 하나씩 차례대로 가져오는 함수
def fetch_data_safely(url):
    try:
        response = requests.get(url, timeout=15) # 15초 대기
        response.raise_for_status() # 접속 에러 확인
        return pd.read_xml(io.BytesIO(response.content), xpath=".//item")
    except Exception as e:
        st.error(f"⚠️ 데이터 노드 접근 실패: {e}")
        return None

@st.cache_data(ttl=3600)
def get_final_data():
    now = datetime.now()
    curr_time = now.strftime("%Y%m%d%H%M")
    start_time = (now - timedelta(days=20)).strftime("%Y%m%d%H%M") # 20일치로 조정
    
    api_key = "5283B235-DEA8-4DC9-9E36-D7A8A9AEA79F"
    wl_url = f"https://api.hrfco.go.kr/{api_key}/waterlevel/list/1D/2004680/{start_time}/{curr_time}.xml"
    rf_url = f"https://api.hrfco.go.kr/{api_key}/rainfall/list/1D/20044080/{start_time}/{curr_time}.xml"
    
    # 1. 수위 데이터 가져오기
    wl_raw = fetch_data_safely(wl_url)
    # 2. 강수량 데이터 가져오기
    rf_raw = fetch_data_safely(rf_url)
    
    if wl_raw is not None and rf_raw is not None:
        wl_df = wl_raw[['ymd', 'fw']]
        rf_df = rf_raw[['ymd', 'rf']]
        df = pd.merge(rf_df, wl_df, on='ymd').rename(columns={'ymd':'날짜', 'rf':'강수량', 'fw':'유량'})
        
        # 무결성 검토
        df[['강수량', '유량']] = df[['강수량', '유량']].apply(pd.to_numeric, errors='coerce')
        df = df.dropna().query("강수량 >= 0 and 유량 >= 0")
        return df.sort_values('날짜')
    return None

if st.button("🔄 데이터 수집 및 예측 실행"):
    with st.spinner("📡 한강홍수통제소 연결 중..."):
        df = get_final_data()
        
        if df is not None and len(df) >= 7:
            # 예측 로직
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(df[['강수량', '유량']].values)
            pred_scaled = model.predict(scaled[-7:].reshape(1, 7, 2), verbose=0)[0]
            
            def to_real(val):
                dummy = np.zeros((1, 2)); dummy[0, 1] = val
                return scaler.inverse_transform(dummy)[0, 1]
            
            # 수치 표시
            st.subheader("📅 예측 결과 (m³/s)")
            c1, c2, c3 = st.columns(3)
            c1.metric("내일", f"{to_real(pred_scaled[0]):.2f}")
            c2.metric("모레", f"{to_real(pred_scaled[1]):.2f}")
            c3.metric("글피", f"{to_real(pred_scaled[2]):.2f}")
            
            st.divider()
            st.table(df.tail(10))
        else:
            st.error("❌ 데이터를 불러올 수 없습니다. API 서버의 일시적 오류일 수 있습니다.")
