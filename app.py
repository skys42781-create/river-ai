import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# [1. 설정 및 모델 로드]
st.set_page_config(page_title="한강 수문 2층 LSTM 분석", layout="wide")
st.title("🌊 120일 시퀀스 & 2층 Stacked LSTM 유량 예측")

# 분석 창 크기 설정 (반드시 학습 시 설정한 look_back과 일치해야 함)
LOOK_BACK = 120 

@st.cache_resource
def get_model():
    # 2층으로 쌓고 120일로 학습한 최신 모델 파일명
    return load_model('river_model_final.h5', compile=False)

try:
    model = get_model()
except Exception as e:
    st.error(f"❌ 모델 로드 실패: {e}. 깃허브에 최신 모델(.h5)이 있는지 확인하세요.")
    st.stop()

# [2. 데이터 로드 및 전처리]
@st.cache_data
def load_river_data():
    csv_url = "https://raw.githubusercontent.com/skys42781-create/river-ai/main/latest_river_data.csv"
    df_raw = pd.read_csv(csv_url)
    df_raw.columns = df_raw.columns.str.strip()
    
    # 컬럼 매핑 (유연한 이름 찾기)
    processed = {}
    mapping = {'강수량': ['rf', '강수'], '유량': ['fw', '유량'], '수위': ['wl', '수위'], '날짜': ['ymd', '날짜', '년월일']}
    for target, keys in mapping.items():
        for col in df_raw.columns:
            if any(k in col.lower() for k in keys):
                if target == '날짜': processed[target] = df_raw[col].astype(str).str[:8]
                else: processed[target] = pd.to_numeric(df_raw[col], errors='coerce')
                break
    
    df = pd.DataFrame(processed)
    # 유량 데이터가 있는 2007년 이후 데이터만 추출 및 정렬
    df = df.dropna(subset=['유량']).query("유량 > 0").sort_values('날짜').reset_index(drop=True)
    return df

df = load_river_data()

# [3. 메인 분석 로직]
if len(df) >= LOOK_BACK + 1:
    # 데이터가 있는 가장 최신 날짜들을 선택지로 제공
    date_options = df['날짜'].tolist()[LOOK_BACK:]
    
    st.sidebar.header("🕒 분석 설정")
    live_mode = st.sidebar.toggle("최신 확정 데이터 자동 추적", value=True)
    selected_t = date_options[-1] if live_mode else st.sidebar.selectbox("기준일(T) 선택", date_options, index=len(date_options)-1)
    
    pos = df[df['날짜'] == selected_t].index[0]

    # --- 구역 1: 실시간 현황판 ---
    st.subheader(f"📊 {selected_t} (T) 시점 수문 요약")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("현재 유량", f"{df['유량'].iloc[pos]:.2f} m³/s")
    m2.metric("현재 수위", f"{df['수위'].iloc[pos]:.2f} m")
    m3.metric("최근 7일 강우", f"{df['강수량'].iloc[pos-6:pos+1].sum():.1f} mm")
    m4.metric("분석 윈도우", f"{LOOK_BACK}일 (4달)")

    # --- 구역 2: 2층 모델 기반 T+1~T+3 예측 ---
    st.divider()
    st.subheader("🔮 향후 3일(T+1~T+3) AI 유량 예측 결과")
    
    scaler = MinMaxScaler()
    # 학습과 동일한 피처 순서 (강수량, 유량)
    scaled = scaler.fit_transform(df[['강수량', '유량']].values)
    input_seq = scaled[pos-(LOOK_BACK-1) : pos+1].reshape(1, LOOK_BACK, 2)
    
    try:
        preds_scaled = model.predict(input_seq, verbose=0)[0]
        def to_real(s_val):
            dummy = np.zeros((1, 2)); dummy[0, 1] = s_val
            return scaler.inverse_transform(dummy)[0, 1]
        
        preds_real = [to_real(p) for p in preds_scaled]
        
        c1, c2, c3 = st.columns(3)
        for i in range(3):
            with [c1, c2, c3][i]:
                st.info(f"**T+{i+1}일 예측**")
                st.title(f"{preds_real[i]:.2f}")
                # 실제값이 존재할 경우 오차 표시
                if pos + i + 1 < len(df):
                    actual = df['유량'].iloc[pos+i+1]
                    err = abs(preds_real[i] - actual) / actual * 100
                    st.write(f"실제: {actual:.2f} (오차 {err:.1f}%)")
    except Exception as e:
        st.error(f"⚠️ 모델 예측 오류: {e}")
        st.info("모델의 입력 크기가 120이 맞는지, 레이어가 2층으로 잘 구성되었는지 확인하세요.")

    # --- 구역 3: [검증] 오차 변동 추이 ---
    st.divider()
    st.subheader("📉 최근 30일간의 모델 예측 정밀도(오차율) 변동")
    
    error_analysis = []
    for p in range(pos-30, pos):
        if p < LOOK_BACK: continue
        t_input = scaled[p-(LOOK_BACK-1) : p+1].reshape(1, LOOK_BACK, 2)
        t_pred = to_real(model.predict(t_input, verbose=0)[0][0]) # T+1 예측값
        t_actual = df['유량'].iloc[p+1]
        error_analysis.append({
            '날짜': df['날짜'].iloc[p+1],
            '오차율(%)': (abs(t_pred - t_actual) / t_actual) * 100
        })
    
    if error_analysis:
        error_df = pd.DataFrame(error_analysis).set_index('날짜')
        st.area_chart(error_df, color="#ff4b4b")

    # --- 구역 4: 시각적 흐름 ---
    st.divider()
    st.subheader(f"📈 최근 {LOOK_BACK}일간의 유량 및 강수량 트렌드")
    st.line_chart(df.iloc[pos-(LOOK_BACK-1) : pos+1].set_index('날짜')[['유량', '강수량']])

else:
    st.error(f"분석을 위한 데이터가 부족합니다. (최소 {LOOK_BACK+1}일 필요)")
