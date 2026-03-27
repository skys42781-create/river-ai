import streamlit as st
import pandas as pd
import numpy as np
import requests
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import io
import time # 재시도 간격 조절을 위해 추가

st.set_page_config(page_title="한강 유량 실시간 예측", layout="wide")
st.title("🌊 한강 유량 실시간 예측 시스템 (네트워크 강화 버전)")

@st.cache_resource
def get_model():
    return load_model('river_model_final.h5', compile=False)

model = get_model()

# [핵심 수정] 타임아웃 연장 및 3회 재시도 로직
def fetch_data_with_retry(url, name):
    max_retries = 3
    for i in range(max_retries):
        try:
            # 타임아웃을 60초로 설정 (기존 15초에서 대폭 연장)
            response = requests.get(url, timeout=60) 
            response.raise_for_status()
            return pd.read_xml(io.BytesIO(response.content), xpath=".//item")
        except Exception as e:
            if i < max_retries - 1:
                st.warning(f"⚠️ {name} 데이터 {i+1}차 시도 실패... 3초 후 다시 시도합니다.")
                time.sleep(3) # 3초 대기 후 재시도
                continue
            else:
                st.error(f"❌ {name} 데이터 최종 수집 실패: {e}")
                return None

@st.cache_data(ttl=3600)
def get_final_data():
    now = datetime.now()
    curr_time = now.strftime("%Y%m%d%H%M")
    start_time = (now - timedelta(days=20)).strftime("%Y%m%d%H%M")
    
    api_key = "5283B235-DEA8-4DC9-9E36-D7A8A9AEA79F"
    wl_url = f"https://api.hrfco.go.kr/{api_key}/waterlevel/list/1D/2004680/{start_time}/{curr_time}.xml"
    rf_url = f"https://api.hrfco.go.kr/{api_key}/rainfall/list/1D/20044080/{start_time}/{curr_time}.xml"
    
    # 순차적으로 재시도 로직 실행
    wl_raw = fetch_data_with_retry(wl_url, "수위/유량")
    if wl_raw is None: return None
    
    rf_raw = fetch_data_with_retry(rf_url, "강수량")
    if rf_raw is None: return None
    
    # 데이터 병합 및 무결성 검토
    wl_df = wl_raw[['ymd', 'fw']]
    rf_df = rf_raw[['ymd', 'rf']]
    df = pd.merge(rf_df, wl_df, on='ymd').rename(columns={'ymd':'날짜', 'rf':'강수량', 'fw':'유량'})
    
    df[['강수량', '유량']] = df[['강수량', '유량']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna().query("강수량 >= 0 and 유량 >= 0")
    return df.sort_values('날짜')

if st.button("🚀 실시간 데이터 수집 및 예측 실행"):
    with st.spinner("⏳ 서버와 연결 중입니다. 최대 3분 정도 소요될 수 있습니다..."):
        df = get_final_data()
        
        if df is not None and len(df) >= 7:
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(df[['강수량', '유량']].values)
            pred_scaled = model.predict(scaled[-7:].reshape(1, 7, 2), verbose=0)[0]
            
            def to_real(val):
                dummy = np.zeros((1, 2)); dummy[0, 1] = val
                return scaler.inverse_transform(dummy)[0, 1]
            
            st.subheader("📅 향후 3일 예측 결과 (m³/s)")
            c1, c2, c3 = st.columns(3)
            c1.metric("내일", f"{to_real(pred_scaled[0]):.2f}")
            c2.metric("모레", f"{to_real(pred_scaled[1]):.2f}")
            c3.metric("글피", f"{to_real(pred_scaled[2]):.2f}")
            
            st.divider()
            st.subheader("📊 최근 관측 데이터 (무결성 확인)")
            st.table(df.tail(10))
        else:
            st.error("❌ 데이터를 가져올 수 없습니다. 아래 '대안 방법'을 확인해 주세요.")
            st.info("💡 만약 지속적으로 실패한다면, 해외 서버 차단일 가능성이 높습니다. 수동 업로드 기능을 사용해 주세요.")
