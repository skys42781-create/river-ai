import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# [1. 기본 설정]
st.set_page_config(page_title="한강 유량 고도화 예측", layout="wide")
st.title("🌊 데이터 가공 기반 유량 예측 시스템 (Look-back: 30D)")
st.markdown("---")

@st.cache_resource
def get_model():
    # 학습 시 look_back을 30으로 설정한 새로운 모델 파일이 필요합니다.
    return load_model('river_model_final.h5', compile=False)

model = get_model()

# [2. 데이터 로드 및 이동 평균(Smoothing) 적용]
@st.cache_data
def load_processed_data():
    csv_url = "https://raw.githubusercontent.com/skys42781-create/river-ai/main/latest_river_data.csv"
    df = pd.read_csv(csv_url)
    
    # 컬럼명 통일
    name_map = {'rf': '강수량', 'fw': '유량', 'ymdhm': '날짜', '년월일(yyyyMMdd)': '날짜'}
    df = df.rename(columns=name_map)
    df[['강수량', '유량']] = df[['강수량', '유량']].apply(pd.to_numeric, errors='coerce')
    df['날짜'] = df['날짜'].astype(str).str[:8]
    df = df.dropna().query("강수량 >= 0 and 유량 >= 0").sort_values('날짜').reset_index(drop=True)

    # --- [데이터 가공: 3일 이동 평균] ---
    # 모델이 너무 예민하게 반응하지 않도록 데이터를 부드럽게 만듭니다.
    df['강수량_SMA'] = df['강수량'].rolling(window=3, min_periods=1).mean()
    df['유량_SMA'] = df['유량'].rolling(window=3, min_periods=1).mean()
    
    return df

df = load_processed_data()

# [3. 메인 분석 화면]
if df is not None and len(df) >= 35: # 30일치 분석을 위해 최소 35일 필요
    # 스케일러 (가공된 데이터를 기준으로 학습했다면 가공 데이터를 넣어야 함)
    scaler = MinMaxScaler()
    # 주의: 학습 시 '강수량_SMA', '유량_SMA'를 썼다면 여기서도 SMA 컬럼을 넣으세요.
    data_for_scale = df[['강수량_SMA', '유량_SMA']].values
    scaled = scaler.fit_transform(data_for_scale)

    st.sidebar.header("📅 분석 시점 선택")
    # 30일 데이터가 확보된 시점부터 선택 가능
    date_list = df['날짜'].tolist()[30:] 
    selected_t = st.sidebar.selectbox("기준일(T) 선택", date_list, index=len(date_list)-1)
    
    pos = df[df['날짜'] == selected_t].index[0]
    
    # [4. 예측 실행: 과거 30일치 입력]
    # (1, 30, 2) 구조로 변경됨
    input_seq = scaled[pos-29 : pos+1].reshape(1, 30, 2)
    preds_scaled = model.predict(input_seq, verbose=0)[0]
    
    def to_real(s_val):
        dummy = np.zeros((1, 2)); dummy[0, 1] = s_val
        return scaler.inverse_transform(dummy)[0, 1]
    
    preds_real = [to_real(p) for p in preds_scaled]

    # [5. 결과 대시보드]
    st.subheader(f"📌 {selected_t} 기준 향후 3일 유량 (30일 흐름 분석 결과)")
    c1, c2, c3 = st.columns(3)
    
    for i in range(3):
        with [c1, c2, c3][i]:
            st.metric(f"T+{i+1}일 예측", f"{preds_real[i]:.2f} m³/s")
            # 사후 검증
            if pos + i + 1 < len(df):
                actual = df['유량'].iloc[pos + i + 1]
                error = abs(preds_real[i] - actual) / actual * 100
                st.write(f"**실제값:** {actual:.2f} | **오차율:** {error:.1f}%")

    # [6. 시각화: 원본 vs 가공 데이터 비교]
    st.divider()
    st.subheader("📈 데이터 가공(Smoothing) 효과 확인")
    st.info("💡 옅은 선은 원본 데이터이고, 굵은 선은 모델이 학습한 3일 이동 평균 데이터입니다.")
    
    # 차트용 데이터 (최근 40일)
    chart_df = df.iloc[max(0, pos-40) : pos+1][['날짜', '유량', '유량_SMA']].set_index('날짜')
    st.line_chart(chart_df)

else:
    st.warning("분석을 위해 최소 35일 이상의 데이터가 필요합니다. get_data.py를 통해 더 많은 데이터를 수집하세요.")
