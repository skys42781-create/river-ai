import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# [1. 설정 및 모델]
st.set_page_config(page_title="한강 수문 정밀 분석", layout="wide")
st.title("🌊 한강 수문 정밀 분석 및 유량 예측 시스템")

@st.cache_resource
def get_model():
    return load_model('river_model_final.h5', compile=False)

model = get_model()

# [2. 데이터 로드: 이름 변경 로직을 최우선으로 배치]
@st.cache_data
def load_rich_data():
    csv_url = "https://raw.githubusercontent.com/skys42781-create/river-ai/main/latest_river_data.csv"
    df = pd.read_csv(csv_url)
    
    # [핵심 수정] 이름을 바꾸는 작업을 가장 먼저 수행합니다.
    name_map = {
        'rf': '강수량', 'rf(mm)': '강수량', '강수량(mm)': '강수량',
        'fw': '유량', 'fw(m3/s)': '유량', '유량(m3/s)': '유량',
        'wl': '수위', 'wl(m)': '수위', '수위(m)': '수위',
        'ymdhm': '날짜', '년월일(yyyyMMdd)': '날짜'
    }
    df = df.rename(columns=name_map)
    
    # 필요한 컬럼이 없으면 에러 방지를 위해 빈 컬럼이라도 생성
    for col in ['강수량', '유량', '수위', '날짜']:
        if col not in df.columns:
            df[col] = 0 if col != '날짜' else "20260101"

    # 숫자형 변환
    df['강수량'] = pd.to_numeric(df['강수량'], errors='coerce').fillna(0)
    df['유량'] = pd.to_numeric(df['유량'], errors='coerce').fillna(0)
    df['수위'] = pd.to_numeric(df['수위'], errors='coerce').fillna(0)
    
    # [지표 생성] 이름이 바뀐 후에 계산을 시작합니다.
    df['강수량_SMA'] = df['강수량'].rolling(window=3, min_periods=1).mean()
    df['유량_SMA'] = df['유량'].rolling(window=3, min_periods=1).mean()
    df['유량변화율'] = df['유량'].pct_change().fillna(0) * 100
    df['누적강수량'] = df['강수량'].rolling(window=7, min_periods=1).sum()

    df['날짜'] = df['날짜'].astype(str).str[:8]
    return df.sort_values('날짜').reset_index(drop=True)

df = load_rich_data()

# [3. 대시보드 출력]
if len(df) >= 31:
    st.sidebar.header("🗓️ 분석 기준일")
    date_options = df['날짜'].tolist()[30:]
    selected_t = st.sidebar.selectbox("T시점 선택", date_options, index=len(date_options)-1)
    pos = df[df['날짜'] == selected_t].index[0]

    # --- 구역 1: Metrics ---
    st.subheader(f"📊 {selected_t} 기준 수문 요약")
    m1, m2, m3, m4 = st.columns(4)
    
    curr_fw = df['유량'].iloc[pos]
    prev_fw = df['유량'].iloc[pos-1] if pos > 0 else curr_fw
    
    m1.metric("현재 유량", f"{curr_fw:.2f} m³/s", f"{curr_fw - prev_fw:.2f}")
    m2.metric("현재 수위", f"{df['수위'].iloc[pos]:.2f} m")
    m3.metric("7일 누적 강우", f"{df['누적강수량'].iloc[pos]:.1f} mm")
    m4.metric("유량 변동성", f"{df['유량변화율'].iloc[pos]:.1f}%")

    # --- 구역 2: AI 예측 ---
    st.divider()
    st.subheader("🔮 향후 3일 인공지능 예측 (30일 분석)")
    
    scaler = MinMaxScaler()
    # 학습 모델이 [강수량_SMA, 유량_SMA]를 사용한다고 가정
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
            st.info(f"**T+{i+1}일 예상**")
            st.title(f"{p:.2f}")
            if pos + i + 1 < len(df):
                act = df['유량'].iloc[pos+i+1]
                st.write(f"실제값: {act:.2f} (오차 {abs(p-act)/act*100:.1f}%)")

    # --- 구역 3: 시각화 ---
    st.divider()
    l, r = st.columns(2)
    with l:
        st.subheader("📈 유량 및 수위 변화")
        st.line_chart(df.iloc[pos-20 : pos+1].set_index('날짜')[['유량', '수위']])
    with r:
        st.subheader("🌧️ 강수 및 누적 강우")
        st.bar_chart(df.iloc[pos-20 : pos+1].set_index('날짜')[['강수량', '누적강수량']])

else:
    st.warning("데이터가 부족합니다 (최소 31일치 필요).")
