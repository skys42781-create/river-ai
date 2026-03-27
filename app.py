import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

st.set_page_config(page_title="한강 유량 실시간 예측", layout="wide")
st.title("🌊 한강홍수통제소 실시간 유량 예측 시스템")

@st.cache_resource
def get_model():
    return load_model('river_model_final.h5', compile=False)

model = get_model()

def fetch_and_validate_data():
    # [핵심 수정] 현재로부터 30일 전까지만 호출하여 타임아웃 방지
    now = datetime.now()
    curr_time = now.strftime("%Y%m%d%H%M")
    start_time = (now - timedelta(days=30)).strftime("%Y%m%d%H%M")
    
    # 수문 관측소 코드 (사용자 제공 번호)
    wl_code = "2004680"
    rf_code = "20044080"
    api_key = "5283B235-DEA8-4DC9-9E36-D7A8A9AEA79F"
    
    wl_url = f"https://api.hrfco.go.kr/{api_key}/waterlevel/list/1D/{wl_code}/{start_time}/{curr_time}.xml"
    rf_url = f"https://api.hrfco.go.kr/{api_key}/rainfall/list/1D/{rf_code}/{start_time}/{curr_time}.xml"
    
    try:
        with st.spinner("📡 최근 30일치 데이터를 수집 중입니다..."):
            # XML 읽기 (timeout 설정 추가)
            wl_df = pd.read_xml(wl_url, xpath=".//item", storage_options={'timeout': 30})[['ymd', 'fw']]
            rf_df = pd.read_xml(rf_url, xpath=".//item", storage_options={'timeout': 30})[['ymd', 'rf']]
            
            df = pd.merge(rf_df, wl_df, on='ymd').rename(columns={
                'ymd': '년월일', 'rf': '강수량(mm)', 'fw': '유량(m3/s)'
            })
            
            # 무결성 검사
            df['강수량(mm)'] = pd.to_numeric(df['강수량(mm)'], errors='coerce')
            df['유량(m3/s)'] = pd.to_numeric(df['유량(m3/s)'], errors='coerce')
            df = df.dropna().loc[(df['강수량(mm)'] >= 0) & (df['유량(m3/s)'] >= 0)]
            
            return df.sort_values('년월일')
    except Exception as e:
        st.error(f"❌ 접속 지연 발생: {e}")
        st.info("💡 정부 API 서버가 혼잡합니다. 잠시 후 다시 버튼을 눌러주세요.")
        return None

if st.button("🔄 최신 데이터 자동 수집 및 예측"):
    df = fetch_and_validate_data()
    
    if df is not None and len(df) >= 7:
        st.success(f"✅ 무결성 검사 완료 ({df['년월일'].iloc[-1]} 기준)")
        
        # 예측 로직 (기존과 동일)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[['강수량(mm)', '유량(m3/s)']].values)
        recent_input = scaled_data[-7:].reshape(1, 7, 2)
        pred_scaled = model.predict(recent_input)[0]
        
        def to_real(val):
            dummy = np.zeros((1, 2))
            dummy[0, 1] = val
            return scaler.inverse_transform(dummy)[0, 1]
        
        st.subheader("📅 향후 3일 예측 결과")
        c1, c2, c3 = st.columns(3)
        c1.metric("내일", f"{to_real(pred_scaled[0]):.2f} m³/s")
        c2.metric("모레", f"{to_real(pred_scaled[1]):.2f} m³/s")
        c3.metric("글피", f"{to_real(pred_scaled[2]):.2f} m³/s")
        
        st.dataframe(df.tail(10))
    else:
        st.warning("데이터가 부족하거나 불러오지 못했습니다.")
