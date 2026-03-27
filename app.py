import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# [1. 기본 설정 및 모델 로드]
st.set_page_config(page_title="한강 수문 1년 주기 분석", layout="wide")
st.title("📅 365일 장기 패턴 기반 유량 예측 시스템")

@st.cache_resource
def get_model():
    # [주의] 깃허브에 있는 실제 모델 파일 이름과 똑같아야 합니다.
    # 만약 파일명이 다르면 'river_model_final.h5' 부분을 수정하세요.
    return load_model('river_model_final.h5', compile=False)

try:
    model = get_model()
except Exception as e:
    st.error(f"❌ 모델 파일을 찾을 수 없습니다: {e}")
    st.stop()

# [2. 데이터 로드 및 고도화 분석]
@st.cache_data
def load_year_data():
    csv_url = "https://raw.githubusercontent.com/skys42781-create/river-ai/main/latest_river_data.csv"
    df = pd.read_csv(csv_url)
    
    # 컬럼명 유연한 매핑
    df.columns = df.columns.str.strip()
    name_map = {
        'rf': '강수량', 'fw': '유량', 'wl': '수위', 'ymdhm': '날짜',
        'rf(mm)': '강수량', 'fw(m3/s)': '유량', 'wl(m)': '수위', '년월일(yyyyMMdd)': '날짜'
    }
    df = df.rename(columns=name_map)
    
    # 필수 수치 변환
    for col in ['강수량', '유량', '수위']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['날짜'] = df['날짜'].astype(str).str[:8]
    
    # 장기 분석용 지표 생성
    df['유량_7D_평균'] = df['유량'].rolling(window=7, min_periods=1).mean()
    df['누적강수_30D'] = df['강수량'].rolling(window=30, min_periods=1).sum()
    df['유량변화율(%)'] = df['유량'].pct_change().fillna(0) * 100

    return df.sort_values('날짜').reset_index(drop=True)

df = load_year_data()

# [3. 365일 분석 및 실시간 T값 설정]
# 1년 분석을 위해서는 최소 366일(365일 과거 데이터 + 오늘)의 데이터가 필요합니다.
if len(df) >= 366:
    date_options = df['날짜'].tolist()[365:]
    
    st.sidebar.header("🕒 분석 모드")
    live_mode = st.sidebar.toggle("실시간 최신 시점 추적", value=True)
    
    if live_mode:
        selected_t = date_options[-1]
        st.sidebar.success(f"현재 실시간 T: {selected_t}")
    else:
        selected_t = st.sidebar.selectbox("과거 날짜 선택", date_options, index=len(date_options)-1)
    
    pos = df[df['날짜'] == selected_t].index[0]

    # --- 구역 1: 연간 요약 Metrics ---
    st.subheader(f"📊 {selected_t} 기준 연간 수문 요약 (T-365D 분석)")
    m1, m2, m3, m4 = st.columns(4)
    
    year_data = df.iloc[pos-364 : pos+1]
    
    m1.metric("현재 유량", f"{df['유량'].iloc[pos]:.2f} m³/s")
    m2.metric("1년 최고 유량", f"{year_data['유량'].max():.2f} m³/s")
    m3.metric("30일 누적 강수", f"{df['누적강수_30D'].iloc[pos]:.1f} mm")
    m4.metric("유량 변동성", f"{df['유량변화율(%)'].iloc[pos]:.1f}%")

    # --- 구역 2: 1년 시퀀스 기반 예측 ---
    st.divider()
    st.subheader("🔮 365일 장기 패턴 기반 AI 예측 (T+1 ~ T+3)")
    
    scaler = MinMaxScaler()
    # [주의] 학습 당시의 피처 순서와 개수를 반드시 맞춰야 합니다.
    scaled = scaler.fit_transform(df[['강수량', '유량']].values)
    
    # (1, 365, 2) 구조로 입력 생성
    input_seq = scaled[pos-364 : pos+1].reshape(1, 365, 2)
    
    try:
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
    except Exception as e:
        st.warning(f"⚠️ 예측 실행 중 오류 발생: {e}")
        st.info("입력 데이터의 길이(365일)가 모델의 설정과 맞지 않을 수 있습니다.")

    # --- 구역 3: 연간 트렌드 차트 ---
    st.divider()
    st.subheader("📈 지난 1년간의 유량 및 강수 흐름")
    st.line_chart(year_data.set_index('날짜')[['유량', '유량_7D_평균']])
    st.bar_chart(year_data.set_index('날짜')['강수량'])

else:
    st.error(f"⚠️ 데이터 부족: 현재 {len(df)}일치 데이터가 있습니다. 365일 분석을 위해 최소 366일치가 필요합니다.")
    st.info("get_data.py의 날짜 범위를 400일 이상으로 수정하여 데이터를 다시 수집하세요.")
