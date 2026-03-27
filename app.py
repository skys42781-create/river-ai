import streamlit as st
import pandas as pd
import numpy as np
import requests
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import io
import time

# [1. 기본 설정]
st.set_page_config(page_title="한강 유량 예측 시스템", layout="wide")
st.title("🌊 한강 유량 실시간 예측 (자동/수동 하이브리드)")

@st.cache_resource
def get_model():
    return load_model('river_model_final.h5', compile=False)

model = get_model()

# [2. API 수집 함수]
def fetch_api_data():
    now = datetime.now()
    curr_time = now.strftime("%Y%m%d%H%M")
    start_time = (now - timedelta(days=20)).strftime("%Y%m%d%H%M")
    api_key = "5283B235-DEA8-4DC9-9E36-D7A8A9AEA79F"
    
    wl_url = f"https://api.hrfco.go.kr/{api_key}/waterlevel/list/1D/2004680/{start_time}/{curr_time}.xml"
    rf_url = f"https://api.hrfco.go.kr/{api_key}/rainfall/list/1D/20044080/{start_time}/{curr_time}.xml"
    
    try:
        with st.spinner("⏳ API 연결 시도 중 (최대 60초)..."):
            wl_res = requests.get(wl_url, timeout=60)
            rf_res = requests.get(rf_url, timeout=60)
            wl_df = pd.read_xml(io.BytesIO(wl_res.content), xpath=".//item")[['ymd', 'fw']]
            rf_df = pd.read_xml(io.BytesIO(rf_res.content), xpath=".//item")[['ymd', 'rf']]
            df = pd.merge(rf_df, wl_df, on='ymd').rename(columns={'ymd':'날짜', 'rf':'강수량', 'fw':'유량'})
            return df
    except:
        return None

# [3. 메인 화면 레이아웃]
st.sidebar.header("🕹️ 컨트롤 패널")
mode = st.sidebar.radio("데이터 입력 방식 선택", ["실시간 API (자동)", "엑셀 파일 업로드 (수동)"])

final_df = None

if mode == "실시간 API (자동)":
    st.info("💡 버튼을 누르면 한강홍수통제소 API에서 최신 자료를 가져옵니다.")
    if st.button("🚀 데이터 자동 수집"):
        final_df = fetch_api_data()
        if final_df is None:
            st.error("❌ API 접속에 실패했습니다. (해외 서버 차단 가능성)")
            st.warning("왼쪽 메뉴에서 '엑셀 파일 업로드' 모드로 변경해 주세요.")

else: # 수동 업로드 모드
    st.info("💡 직접 수집한 강수량/유량 엑셀 파일을 업로드하세요.")
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        rain_file = st.file_uploader("강수량 엑셀", type=['xlsx'])
    with col_up2:
        flow_file = st.file_uploader("유량 엑셀", type=['xlsx'])
    
    if rain_file and flow_file:
        r_df = pd.read_excel(rain_file)
        f_df = pd.read_excel(flow_file)
        # 컬럼명 표준화 (무결성 검토용)
        final_df = pd.merge(r_df, f_df, on='년월일(yyyyMMdd)').rename(columns={'년월일(yyyyMMdd)':'날짜', '강수량(mm)':'강수량', '유량(m3/s)':'유량'})

# [4. 예측 실행 로직 (공통)]
if final_df is not None:
    # --- 무결성 검토 ---
    final_df[['강수량', '유량']] = final_df[['강수량', '유량']].apply(pd.to_numeric, errors='coerce')
    final_df = final_df.dropna().query("강수량 >= 0 and 유량 >= 0").sort_values('날짜')
    
    if len(final_df) >= 7:
        st.success(f"✅ 데이터 무결성 검토 완료 ({final_df['날짜'].iloc[-1]} 기준)")
        
        # 예측 수행
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(final_df[['강수량', '유량']].values)
        pred = model.predict(scaled[-7:].reshape(1, 7, 2), verbose=0)[0]
        
        def to_real(val):
            dummy = np.zeros((1, 2)); dummy[0, 1] = val
            return scaler.inverse_transform(dummy)[0, 1]
        
        # 결과 대시보드
        st.subheader("📅 향후 3일 유량 예측")
        c1, c2, c3 = st.columns(3)
        c1.metric("내일 (T+1)", f"{to_real(pred[0]):.2f} m³/s")
        c2.metric("모레 (T+2)", f"{to_real(pred[1]):.2f} m³/s")
        c3.metric("글피 (T+3)", f"{to_real(pred[2]):.2f} m³/s")
        
        st.divider()
        st.dataframe(final_df.tail(10), use_container_width=True)
    else:
        st.error("데이터가 부족합니다. 최소 7일 이상의 자료가 필요합니다.")
