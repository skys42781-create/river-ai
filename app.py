import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# [1. 설정 및 모델 로드]
st.set_page_config(page_title="한강 수문 1년 분석 시스템", layout="wide")
st.title("📅 365일 장기 패턴 기반 유량 분석 및 예측")

@st.cache_resource
def get_model():
    # 저장소의 실제 모델 파일명과 일치해야 합니다. (기본값: river_model_final.h5)
    return load_model('river_model_final.h5', compile=False)

try:
    model = get_model()
except Exception as e:
    st.error(f"❌ 모델 로드 실패: {e}. 깃허브에 .h5 파일이 있는지 확인하세요.")
    st.stop()

# [2. 데이터 로드: 초강력 이름 찾기 로직]
@st.cache_data
def load_year_data():
    csv_url = "https://raw.githubusercontent.com/skys42781-create/river-ai/main/latest_river_data.csv"
    df = pd.read_csv(csv_url)
    
    # 모든 컬럼명의 앞뒤 공백 제거
    df.columns = df.columns.str.strip()
    
    # [방어 로직] 특정 단어가 포함된 컬럼을 찾아서 강제로 변경
    new_cols = {}
    for col in df.columns:
        c_lower = col.lower()
        if 'rf' in c_lower or '강수' in c_lower:
            new_cols[col] = '강수량'
        elif 'fw' in c_lower or '유량' in c_lower:
            new_cols[col] = '유량'
        elif 'wl' in c_lower or '수위' in c_lower:
            new_cols[col] = '수위'
        elif 'ymd' in c_lower or '날짜' in c_lower or '년월일' in c_lower:
            new_cols[col] = '날짜'
            
    df = df.rename(columns=new_cols)
    
    # 필수 컬럼 존재 여부 최종 확인 및 생성
    for col in ['강수량', '유량', '수위', '날짜']:
        if col not in df.columns:
            df[col] = 0 if col != '날짜' else "20260101"

    # 수치형 변환
    df['강수량'] = pd.to_numeric(df['강수량'], errors='coerce').fillna(0)
    df['유량'] = pd.to_numeric(df['유량'], errors='coerce').fillna(0)
    df['수위'] = pd.to_numeric(df['수위'], errors='coerce').fillna(0)
    df['날짜'] = df['날짜'].astype(str).str[:8]
    
    # 이동 평균 및 통계 (이름이 바뀐 '후'에 계산하므로 에러 없음)
    df['유량_7D_평균'] = df['유량'].rolling(window=7, min_periods=1).mean()
    df['누적강수_30D'] = df['강수량'].rolling(window=30, min_periods=1).sum()
    
    return df.sort_values('날짜').reset_index(drop=True)

df = load_year_data()

# [3. 365일 분석 및 T값 설정]
# 365일치를 보려면 데이터가 최소 366일 이상 있어야 합니다.
if len(df) >= 366:
    date_options = df['날짜'].tolist()[365:]
    
    st.sidebar.header("🕒 분석 시점")
    live_mode = st.sidebar.toggle("최신 시점 자동 추적", value=True)
    selected_t = date_options[-1] if live_mode else st.sidebar.selectbox("과거 날짜 선택", date_options, index=len(date_options)-1)
    
    pos = df[df['날짜'] == selected_t].index[0]

    # --- 구역 1: 통계 카드 ---
    st.subheader(f"📊 {selected_t} 기준 수문 분석 (T-365D)")
    m1, m2, m3, m4 = st.columns(4)
    year_slice = df.iloc[pos-364 : pos+1]
    
    m1.metric("현재 유량", f"{df['유량'].iloc[pos]:.2f} m³/s")
    m2.metric("연간 최고 유량", f"{year_slice['유량'].max():.2f} m³/s")
    m3.metric("최근 30일 강우", f"{df['누적강수_30D'].iloc[pos]:.1f} mm")
    m4.metric("현재 수위", f"{df['수위'].iloc[pos]:.2f} m")

    # --- 구역 2: 365일 기반 AI 예측 ---
    st.divider()
    st.subheader("🔮 365일 장기 시퀀스 분석 기반 예측 (T+1 ~ T+3)")
    
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['강수량', '유량']].values)
    
    # 365일치 입력 (1, 365, 2)
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
        st.warning(f"⚠️ 예측 실패: {e}")
        st.info("입력 데이터의 길이(365)가 모델 학습 당시의 길이와 다를 수 있습니다.")

    # --- 구역 3: 그래프 ---
    st.divider()
    st.subheader("📈 지난 1년간의 수문 변화 트렌드")
    st.line_chart(year_slice.set_index('날짜')[['유량', '유량_7D_평균']])
    st.bar_chart(year_slice.set_index('날짜')['강수량'])

else:
    st.error(f"⚠️ 데이터 부족: 현재 {len(df)}일치 데이터가 있습니다. 1년(365일) 분석을 위해서는 최소 366일치가 필요합니다.")
    st.info("get_data.py를 수정하여 더 넓은 범위(예: 20250101 ~ 20260328)의 데이터를 수집하세요.")
