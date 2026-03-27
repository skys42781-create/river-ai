import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# [1. 설정 및 모델]
st.set_page_config(page_title="한강 수문 실시간 분석", layout="wide")
st.title("🌊 실시간 한강 유량 예측 및 모니터링 시스템")

@st.cache_resource
def get_model():
    return load_model('river_model_final.h5', compile=False)

model = get_model()

# [2. 데이터 로드]
@st.cache_data
def load_live_data():
    csv_url = "https://raw.githubusercontent.com/skys42781-create/river-ai/main/latest_river_data.csv"
    df = pd.read_csv(csv_url)
    
    # 이름 통일
    name_map = {
        'rf': '강수량', 'fw': '유량', 'wl': '수위', 'ymdhm': '날짜',
        'rf_smooth': '강수량_SMA', 'fw_smooth': '유량_SMA'
    }
    df = df.rename(columns=name_map)
    
    # 숫자형 변환 및 날짜 정리
    df['강수량'] = pd.to_numeric(df['강수량'], errors='coerce').fillna(0)
    df['유량'] = pd.to_numeric(df['유량'], errors='coerce').fillna(0)
    df['수위'] = pd.to_numeric(df['수위'], errors='coerce').fillna(0)
    df['날짜'] = df['날짜'].astype(str).str[:8]
    
    # 가공 지표 생성
    df['강수량_SMA'] = df['강수량'].rolling(window=3, min_periods=1).mean()
    df['유량_SMA'] = df['유량'].rolling(window=3, min_periods=1).mean()
    df['누적강수량'] = df['강수량'].rolling(window=7, min_periods=1).sum()
    df['유량변화율'] = df['유량'].pct_change().fillna(0) * 100

    return df.sort_values('날짜').reset_index(drop=True)

df = load_live_data()

# [3. T값 자동 설정 및 분석]
if len(df) >= 31:
    # --- 핵심 변경 부분: 기본값을 마지막 날짜(최신)로 설정 ---
    date_options = df['날짜'].tolist()[30:]
    
    st.sidebar.header("🕒 시점 모드")
    mode = st.sidebar.radio("모드 선택", ["실시간 (최신 날짜)", "과거 시점 분석"])
    
    if mode == "실시간 (최신 날짜)":
        # 리스트의 맨 마지막 날짜를 자동으로 선택
        selected_t = date_options[-1]
        st.sidebar.success(f"현재 기준일: {selected_t}")
    else:
        selected_t = st.sidebar.selectbox("과거 날짜 선택", date_options, index=len(date_options)-1)
    
    pos = df[df['날짜'] == selected_t].index[0]

    # --- 구역 1: 실시간 현황 (Metrics) ---
    st.subheader(f"📍 기준 시점(T): {selected_t} 현황")
    m1, m2, m3, m4 = st.columns(4)
    
    curr_fw = df['유량'].iloc[pos]
    prev_fw = df['유량'].iloc[pos-1] if pos > 0 else curr_fw
    
    m1.metric("현재 유량", f"{curr_fw:.2f} m³/s", f"{curr_fw - prev_fw:.2f}")
    m2.metric("현재 수위", f"{df['수위'].iloc[pos]:.2f} m")
    m3.metric("7일 누적 강우", f"{df['누적강수량'].iloc[pos]:.1f} mm")
    m4.metric("전일 대비 변동", f"{df['유량변화율'].iloc[pos]:.1f}%")

    # --- 구역 2: 미래 예측 (T+1, T+2, T+3) ---
    st.divider()
    st.subheader("🔮 향후 3일 인공지능 예측 결과")
    
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['강수량_SMA', '유량_SMA']].values)
    input_seq = scaled[pos-29 : pos+1].reshape(1, 30, 2)
    preds_scaled = model.predict(input_seq, verbose=0)[0]
    
    def to_real(s_val):
        dummy = np.zeros((1, 2)); dummy[0, 1] = s_val
        return scaler.inverse_transform(dummy)[0, 1]
    
    preds = [to_real(p) for p in preds_scaled]
    
    c1, c2, c3 = st.columns(3)
    for i, p in enumerate(preds):
        with [c1, c2, c3][i]:
            st.info(f"**T+{i+1}일 ({int(selected_t)+i+1 if i < 2 else '미래'})**")
            st.title(f"{p:.2f}")
            st.caption("단위: m³/s")

    # --- 구역 3: 트렌드 차트 ---
    st.divider()
    st.subheader("📈 최근 20일 수문 데이터 트렌드")
    st.line_chart(df.iloc[pos-20 : pos+1].set_index('날짜')[['유량', '수위']])

else:
    st.warning("데이터가 부족합니다.")
