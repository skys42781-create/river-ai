import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# [1. 설정 및 모델 로드]
st.set_page_config(page_title="한강 수문 1년 주기 정밀 분석", layout="wide")
st.title("📅 365일 빅데이터 기반 유량 예측 시스템")

@st.cache_resource
def get_model():
    # 깃허브에 올린 모델 파일명과 일치해야 합니다. (기본값: river_model_final.h5)
    return load_model('river_model_final.h5', compile=False)

try:
    model = get_model()
except Exception as e:
    st.error(f"❌ 모델 로드 실패: {e}")
    st.stop()

# [2. 데이터 로드 및 정제: 공백 데이터 자동 처리]
@st.cache_data
def load_master_data():
    csv_url = "https://raw.githubusercontent.com/skys42781-create/river-ai/main/latest_river_data.csv"
    df_raw = pd.read_csv(csv_url)
    df_raw.columns = df_raw.columns.str.strip()
    
    # 컬럼 매핑 (중복 방지)
    processed = {}
    mapping = {'강수량': ['rf', '강수'], '유량': ['fw', '유량'], '수위': ['wl', '수위'], '날짜': ['ymd', '날짜', '년월일']}
    
    for target, keys in mapping.items():
        for col in df_raw.columns:
            if any(k in col.lower() for k in keys):
                if target == '날짜':
                    processed[target] = df_raw[col].astype(str).str[:8]
                else:
                    processed[target] = pd.to_numeric(df_raw[col], errors='coerce')
                break
    
    df = pd.DataFrame(processed)
    
    # [핵심] 유량 데이터가 존재하는 2007년 이후 데이터만 추출
    df = df.dropna(subset=['유량']).query("유량 > 0").sort_values('날짜').reset_index(drop=True)
    
    # 분석 지표 생성
    df['유량_7D_평균'] = df['유량'].rolling(window=7, min_periods=1).mean()
    df['누적강수_30D'] = df['강수량'].rolling(window=30, min_periods=1).sum()
    
    return df

df = load_master_data()

# [3. 분석 및 예측 로직]
if len(df) >= 366:
    # 데이터가 존재하는 가장 마지막 날짜를 'T'로 설정 (최신 공백 자동 회피)
    date_options = df['날짜'].tolist()[365:]
    
    st.sidebar.header("🕒 분석 시점 설정")
    live_mode = st.sidebar.toggle("유량 데이터가 있는 최신일 자동 추적", value=True)
    selected_t = date_options[-1] if live_mode else st.sidebar.selectbox("날짜 선택", date_options, index=len(date_options)-1)
    
    pos = df[df['날짜'] == selected_t].index[0]
    year_slice = df.iloc[pos-364 : pos+1]

    # --- 구역 1: 주요 요약 (Metrics) ---
    st.subheader(f"📊 {selected_t} (T) 기준 분석 현황")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("현재 유량", f"{df['유량'].iloc[pos]:.2f} m³/s")
    m2.metric("연간 최고 유량", f"{year_slice['유량'].max():.2f} m³/s")
    m3.metric("최근 30일 강우", f"{df['누적강수_30D'].iloc[pos]:.1f} mm")
    m4.metric("현재 수위", f"{df['수위'].iloc[pos]:.2f} m")

    # --- 구역 2: 365일 기반 AI 예측 ---
    st.divider()
    st.subheader("🔮 365일(1년) 시퀀스 분석 기반 예측 (T+1 ~ T+3)")
    
    scaler = MinMaxScaler()
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
                st.caption("단위: m³/s")
    except Exception as e:
        st.error(f"⚠️ 모델 구조 불일치: {e}")
        st.info("현재 365일치를 입력 중입니다. 모델이 7일용으로 학습되었다면 재학습이 필요합니다.")

    # --- 구역 3: 시각화 ---
    st.divider()
    st.subheader("📈 지난 1년간의 유량 변동 추이")
    st.line_chart(year_slice.set_index('날짜')[['유량', '유량_7D_평균']])
    st.caption("※ 굵은 선은 하천의 기저 흐름을 파악하기 위한 7일 이동 평균선입니다.")

else:
    st.error(f"⚠️ 데이터 부족: 1년 분석을 위해서는 최소 366일의 유량 데이터가 필요합니다. (현재 {len(df)}일 확보)")
