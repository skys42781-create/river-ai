import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# [1. 기본 설정 및 모델]
st.set_page_config(page_title="한강 수문 실시간 분석", layout="wide")
st.title("🌊 실시간 한강 유량 예측 및 모니터링 시스템")

@st.cache_resource
def get_model():
    return load_model('river_model_final.h5', compile=False)

model = get_model()

# [2. 데이터 로드: 무적의 컬럼 매핑]
@st.cache_data
def load_live_data():
    csv_url = "https://raw.githubusercontent.com/skys42781-create/river-ai/main/latest_river_data.csv"
    df = pd.read_csv(csv_url)
    
    # 1단계: 컬럼명의 공백 제거 (의외로 공백 때문에 에러가 자주 납니다)
    df.columns = df.columns.str.strip()
    
    # 2단계: 어떤 이름이든 강제로 우리가 쓸 이름으로 통역
    translation = {
        'rf': '강수량', 'rf(mm)': '강수량', '강수량(mm)': '강수량', 'Rainfall': '강수량',
        'fw': '유량', 'fw(m3/s)': '유량', '유량(m3/s)': '유량', 'Flow': '유량',
        'wl': '수위', 'wl(m)': '수위', '수위(m)': '수위', 'WaterLevel': '수위',
        'ymdhm': '날짜', '년월일(yyyyMMdd)': '날짜', 'Date': '날짜'
    }
    df = df.rename(columns=translation)
    
    # 3단계: 필수 컬럼이 없으면 0으로 채워서 생성 (KeyError 방지)
    for col in ['강수량', '유량', '수위']:
        if col not in df.columns:
            df[col] = 0
            
    # 숫자형 변환 및 데이터 정리
    df['강수량'] = pd.to_numeric(df['강수량'], errors='coerce').fillna(0)
    df['유량'] = pd.to_numeric(df['유량'], errors='coerce').fillna(0)
    df['수위'] = pd.to_numeric(df['수위'], errors='coerce').fillna(0)
    df['날짜'] = df['날짜'].astype(str).str[:8]
    
    # 이동 평균 및 지표 계산
    df['강수량_SMA'] = df['강수량'].rolling(window=3, min_periods=1).mean()
    df['유량_SMA'] = df['유량'].rolling(window=3, min_periods=1).mean()
    df['누적강수량'] = df['강수량'].rolling(window=7, min_periods=1).sum()
    df['유량변화율'] = df['유량'].pct_change().fillna(0) * 100

    return df.sort_values('날짜').reset_index(drop=True)

df = load_live_data()

# [3. 실시간 T값 자동 설정 및 분석]
if len(df) >= 31:
    # 데이터 중 가장 최신 날짜를 'T'로 자동 선택
    date_options = df['날짜'].tolist()[30:]
    
    st.sidebar.header("🕒 분석 모드 설정")
    # 사용자가 접속할 때마다 '실시간 모드'가 기본값이 됩니다.
    live_mode = st.sidebar.toggle("실시간 T값 추적", value=True)
    
    if live_mode:
        selected_t = date_options[-1] # 가장 마지막(최신) 데이터
        st.sidebar.info(f"📍 현재 실시간 T: {selected_t}")
    else:
        selected_t = st.sidebar.selectbox("분석 기준일(T) 직접 선택", date_options, index=len(date_options)-1)
    
    pos = df[df['날짜'] == selected_t].index[0]

    # --- 구역 1: Metrics 현황판 ---
    st.subheader(f"📊 {selected_t} (T) 기준 실시간 수문 분석")
    m1, m2, m3, m4 = st.columns(4)
    
    curr_fw = df['유량'].iloc[pos]
    prev_fw = df['유량'].iloc[pos-1] if pos > 0 else curr_fw
    
    m1.metric("현재 유량", f"{curr_fw:.2f} m³/s", f"{curr_fw - prev_fw:.2f}")
    m2.metric("현재 수위", f"{df['수위'].iloc[pos]:.2f} m")
    m3.metric("7일 누적 강우", f"{df['누적강수량'].iloc[pos]:.1f} mm")
    m4.metric("유량 변동성", f"{df['유량변화율'].iloc[pos]:.1f}%")

    # --- 구역 2: 미래 예측 결과 ---
    st.divider()
    st.subheader("🔮 향후 3일(T+1~T+3) AI 유량 예측")
    
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
            st.info(f"**T+{i+1}일 예측**")
            st.title(f"{p:.2f}")
            st.caption("m³/s")

    # --- 구역 3: 트렌드 시각화 ---
    st.divider()
    l, r = st.columns(2)
    with l:
        st.subheader("📈 유량 및 수위 변화 (최근 20일)")
        st.line_chart(df.iloc[pos-20 : pos+1].set_index('날짜')[['유량', '수위']])
    with r:
        st.subheader("🌧️ 강수량 데이터 분석")
        st.bar_chart(df.iloc[pos-20 : pos+1].set_index('날짜')[['강수량', '누적강수량']])

else:
    st.error("데이터가 부족합니다. 최소 31일 이상의 자료가 필요합니다.")
