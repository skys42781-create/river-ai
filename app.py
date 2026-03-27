import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# [1. 설정]
st.set_page_config(page_title="한강 수문 연간 분석 시스템", layout="wide")
st.title("📅 1년(365D) 장기 패턴 기반 유량 예측")

@st.cache_resource
def get_model():
    # 365일용 대용량 모델 로드
    return load_model('river_model_365d.h5', compile=False)

model = get_model()

# [2. 대용량 데이터 로드]
@st.cache_data
def load_long_term_data():
    csv_url = "https://raw.githubusercontent.com/skys42781-create/river-ai/main/latest_river_data.csv"
    df = pd.read_csv(csv_url)
    
    # 컬럼 매핑 및 전처리 (이전과 동일하지만 대용량 데이터 처리)
    name_map = {'rf': '강수량', 'fw': '유량', 'wl': '수위', 'ymdhm': '날짜'}
    df = df.rename(columns=name_map)
    df.columns = df.columns.str.strip()
    
    # 필수 수치 변환
    for col in ['강수량', '유량', '수위']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['날짜'] = df['날짜'].astype(str).str[:8]
    
    # 1년 장기 분석을 위한 이동 평균 (7일/30일 SMA 추가)
    df['유량_7D_Avg'] = df['유량'].rolling(window=7, min_periods=1).mean()
    df['강수_30D_Sum'] = df['강수량'].rolling(window=30, min_periods=1).sum()
    
    return df.sort_values('날짜').reset_index(drop=True)

df = load_long_term_data()

# [3. 365일 분석 로직]
if len(df) >= 366: # 최소 1년치 데이터가 있어야 함
    date_options = df['날짜'].tolist()[365:]
    selected_t = st.sidebar.selectbox("기준일(T) 선택", date_options, index=len(date_options)-1)
    pos = df[df['날짜'] == selected_t].index[0]

    # --- 구역 1: 연간 요약 지표 ---
    st.subheader(f"📊 {selected_t} 기준 연간 수문 통계")
    m1, m2, m3 = st.columns(3)
    
    # 지난 1년간의 데이터 추출
    last_year_data = df.iloc[pos-364 : pos+1]
    
    m1.metric("1년 최고 유량", f"{last_year_data['유량'].max():.2f} m³/s")
    m2.metric("1년 평균 유량", f"{last_year_data['유량'].mean():.2f} m³/s")
    m3.metric("연간 누적 강수", f"{last_year_data['강수량'].sum():.1f} mm")

    # --- 구역 2: 365일 기반 예측 ---
    st.divider()
    st.subheader("🔮 365일 장기 시퀀스 분석 결과 (T+1 ~ T+3)")
    
    scaler = MinMaxScaler()
    # 입력 피처: 강수량, 유량 (365일치)
    scaled = scaler.fit_transform(df[['강수량', '유량']].values)
    
    # (1, 365, 2) 형태로 모델에 주입
    input_seq = scaled[pos-364 : pos+1].reshape(1, 365, 2)
    preds_scaled = model.predict(input_seq, verbose=0)[0]
    
    def to_real(s_val):
        dummy = np.zeros((1, 2)); dummy[0, 1] = s_val
        return scaler.inverse_transform(dummy)[0, 1]
    
    preds = [to_real(p) for p in preds_scaled]
    
    c1, c2, c3 = st.columns(3)
    for i, p in enumerate(preds):
        with [c1, c2, c3][i]:
            st.metric(f"T+{i+1}일 예측", f"{p:.2f} m³/s")

    # --- 구역 3: 연간 그래프 (시각화의 꽃) ---
    st.divider()
    st.subheader("📈 지난 1년간의 수문 변화 추이")
    # 전체 1년치를 보여주면 데이터가 너무 빽빽하므로 주 단위나 월 단위 흐름을 보여주는 게 좋습니다.
    st.line_chart(last_year_data.set_index('날짜')[['유량', '유량_7D_Avg']])

else:
    st.error(f"데이터가 부족합니다. 현재 {len(df)}일치 데이터가 있습니다. 1년 분석을 위해 최소 366일치가 필요합니다.")
