import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# [1. 설정 및 모델]
st.set_page_config(page_title="한강 수문 정밀 분석 대시보드", layout="wide")
st.title("🌊 한강 수문 정밀 분석 및 유량 예측 시스템")

@st.cache_resource
def get_model():
    return load_model('river_model_final.h5', compile=False)

model = get_model()

# [2. 데이터 로드 및 다각도 분석]
@st.cache_data
def load_rich_data():
    csv_url = "https://raw.githubusercontent.com/skys42781-create/river-ai/main/latest_river_data.csv"
    df = pd.read_csv(csv_url)
    
    # 컬럼 매핑 (유연한 대응)
    name_map = {
        'rf': '강수량', 'fw': '유량', 'wl': '수위', 'ymdhm': '날짜',
        'rf_smooth': '강수량_SMA', 'fw_smooth': '유량_SMA'
    }
    df = df.rename(columns=name_map)
    
    # 숫자형 변환
    cols = ['강수량', '유량', '수위']
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # [새로운 지표 생성]
    # 1. 3일 이동 평균 (Smoothing)
    df['강수량_SMA'] = df['강수량'].rolling(window=3).mean()
    df['유량_SMA'] = df['유량'].rolling(window=3).mean()
    # 2. 전일 대비 유량 변화율 (%)
    df['유량변화율'] = df['유량'].pct_change() * 100
    # 3. 최근 7일 누적 강수량 (토양 수분 지표 대용)
    df['누적강수량'] = df['강수량'].rolling(window=7).sum()

    df['날짜'] = df['날짜'].astype(str).str[:8]
    return df.dropna(subset=['강수량', '유량']).sort_values('날짜').reset_index(drop=True)

df = load_rich_data()

# [3. 대시보드 레이아웃]
if len(df) >= 31:
    st.sidebar.header("🗓️ 분석 기준일")
    selected_t = st.sidebar.selectbox("T시점 선택", df['날짜'].tolist()[30:], index=len(df)-32)
    pos = df[df['날짜'] == selected_t].index[0]

    # --- 구역 1: 주요 요약 지표 (Metrics) ---
    st.subheader(f"📊 {selected_t} 기준 수문 요약")
    m1, m2, m3, m4 = st.columns(4)
    
    curr_fw = df['유량'].iloc[pos]
    prev_fw = df['유량'].iloc[pos-1]
    curr_rf_7 = df['누적강수량'].iloc[pos]
    curr_wl = df['수위'].iloc[pos] if '수위' in df.columns else 0

    m1.metric("현재 유량", f"{curr_fw:.2f} m³/s", f"{curr_fw - prev_fw:.2f}")
    m2.metric("현재 수위", f"{curr_wl:.2f} m", help="관측소 기준 하천 수위")
    m3.metric("7일 누적 강우", f"{curr_rf_7:.1f} mm", "최근 1주일 합계")
    m4.metric("유량 변동성", f"{df['유량변화율'].iloc[pos]:.1f}%", "전일 대비")

    # --- 구역 2: 미래 예측 결과 ---
    st.divider()
    st.subheader("🔮 향후 3일 인공지능 예측")
    
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
            st.info(f"**T+{i+1}일 예상**")
            st.title(f"{p:.2f}")
            st.caption("단위: m³/s")

    # --- 구역 3: 다각도 시각화 (Charts) ---
    st.divider()
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📈 유량 및 수위 상관관계")
        # 이중 축 느낌으로 유량과 수위를 함께 보여줌
        st.line_chart(df.iloc[pos-20 : pos+1].set_index('날짜')[['유량', '수위']])
    
    with col_right:
        st.subheader("🌧️ 강수량 및 누적 강수 흐름")
        st.bar_chart(df.iloc[pos-20 : pos+1].set_index('날짜')[['강수량', '누적강수량']])

    # --- 구역 4: 데이터 상세 테이블 ---
    with st.expander("🔍 상세 데이터 로그 확인"):
        st.write(df.iloc[pos-10 : pos+1])

else:
    st.warning("데이터를 분석 중입니다. 최소 31일 이상의 자료가 필요합니다.")
