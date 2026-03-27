import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# [1. 설정 및 모델]
st.set_page_config(page_title="한강 수문 1년 분석 시스템", layout="wide")
st.title("📅 365일 장기 패턴 기반 유량 예측")

@st.cache_resource
def get_model():
    # 깃허브의 실제 모델 파일명과 일치해야 합니다.
    return load_model('river_model_final.h5', compile=False)

try:
    model = get_model()
except Exception as e:
    st.error(f"❌ 모델 로드 실패: {e}")
    st.stop()

# [2. 데이터 로드: 중복 컬럼 방지 로직]
@st.cache_data
def load_year_data():
    csv_url = "https://raw.githubusercontent.com/skys42781-create/river-ai/main/latest_river_data.csv"
    df_raw = pd.read_csv(csv_url)
    df_raw.columns = df_raw.columns.str.strip()
    
    # [핵심 수정] 중복 방지를 위해 새로운 데이터프레임을 생성합니다.
    processed_data = {}
    
    # 키워드 매핑 (가장 먼저 발견되는 컬럼 하나만 선택)
    mapping = {
        '강수량': ['rf', '강수'],
        '유량': ['fw', '유량'],
        '수위': ['wl', '수위'],
        '날짜': ['ymd', '날짜', '년월일']
    }
    
    for target, keywords in mapping.items():
        for col in df_raw.columns:
            if any(k in col.lower() for k in keywords):
                # 발견 즉시 numeric 변환 후 1차원 데이터로 저장
                series = pd.to_numeric(df_raw[col], errors='coerce').fillna(0)
                if target == '날짜':
                    processed_data[target] = df_raw[col].astype(str).str[:8]
                else:
                    processed_data[target] = series
                break # 하나 찾으면 다음 키워드로 넘어감 (중복 방지)

    df = pd.DataFrame(processed_data)
    
    # 필수 컬럼 보장
    for col in ['강수량', '유량', '수위', '날짜']:
        if col not in df.columns:
            df[col] = 0 if col != '날짜' else "20260101"

    # 이동 평균 계산
    df['유량_7D_평균'] = df['유량'].rolling(window=7, min_periods=1).mean()
    df['누적강수_30D'] = df['강수량'].rolling(window=30, min_periods=1).sum()
    
    return df.sort_values('날짜').reset_index(drop=True)

df = load_year_data()

# [3. 대시보드 출력]
if len(df) >= 366:
    date_options = df['날짜'].tolist()[365:]
    
    st.sidebar.header("🕒 분석 시점")
    live_mode = st.sidebar.toggle("최신 시점 자동 추적", value=True)
    selected_t = date_options[-1] if live_mode else st.sidebar.selectbox("날짜 선택", date_options, index=len(date_options)-1)
    
    pos = df[df['날짜'] == selected_t].index[0]

    # --- Metrics ---
    st.subheader(f"📊 {selected_t} 기준 분석")
    m1, m2, m3, m4 = st.columns(4)
    year_slice = df.iloc[pos-364 : pos+1]
    
    m1.metric("현재 유량", f"{df['유량'].iloc[pos]:.2f} m³/s")
    m2.metric("연간 최고 유량", f"{year_slice['유량'].max():.2f} m³/s")
    m3.metric("최근 30일 강우", f"{df['누적강수_30D'].iloc[pos]:.1f} mm")
    m4.metric("현재 수위", f"{df['수위'].iloc[pos]:.2f} m")

    # --- AI 예측 ---
    st.divider()
    st.subheader("🔮 365일 기반 AI 예측 (T+1 ~ T+3)")
    
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['강수량', '유량']].values)
    
    # 입력 시퀀스 생성 (1, 365, 2)
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
    except Exception as e:
        st.error(f"⚠️ 모델 구조 오류: {e}")
        st.info("현재 365일 데이터를 넣고 있는데, 모델이 7일용일 수 있습니다. 'look_back' 설정을 확인하세요.")

    # --- 차트 ---
    st.divider()
    st.line_chart(year_slice.set_index('날짜')[['유량', '유량_7D_평균']])
    st.bar_chart(year_slice.set_index('날짜')['강수량'])

else:
    st.error(f"⚠️ 데이터 부족: 현재 {len(df)}일치 데이터가 있습니다. 365일 분석을 위해 최소 366일치가 필요합니다.")
