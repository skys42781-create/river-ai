import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# [1. 기본 설정]
st.set_page_config(page_title="한강 유량 고도화 예측", layout="wide")
st.title("🌊 데이터 가공 기반 유량 예측 시스템 (Look-back: 30D)")

@st.cache_resource
def get_model():
    return load_model('river_model_final.h5', compile=False)

model = get_model()

# [2. 데이터 로드: 에러 방지 강화 버전]
@st.cache_data
def load_processed_data():
    csv_url = "https://raw.githubusercontent.com/skys42781-create/river-ai/main/latest_river_data.csv"
    df = pd.read_csv(csv_url)
    
    # [수정 포인트] 파일에 적힌 이름을 강제로 통역합니다.
    # 파일 헤더가 'rf'든 '강수량(mm)'이든 상관없이 바꿉니다.
    translation = {
        'rf': '강수량', 'rf(mm)': '강수량', '강수량(mm)': '강수량', 'rf_smooth': '강수량_SMA',
        'fw': '유량', 'fw(m3/s)': '유량', '유량(m3/s)': '유량', 'fw_smooth': '유량_SMA',
        'ymdhm': '날짜', '년월일(yyyyMMdd)': '날짜', '날짜': '날짜'
    }
    
    # 존재하는 컬럼만 골라서 이름을 변경합니다.
    df = df.rename(columns=translation)
    
    # 만약 get_data.py에서 SMA(이동평균)를 안 만들었다면 여기서 직접 만듭니다.
    if '강수량' in df.columns and '강수량_SMA' not in df.columns:
        df['강수량_SMA'] = pd.to_numeric(df['강수량'], errors='coerce').rolling(window=3, min_periods=1).mean()
    if '유량' in df.columns and '유량_SMA' not in df.columns:
        df['유량_SMA'] = pd.to_numeric(df['유량'], errors='coerce').rolling(window=3, min_periods=1).mean()

    # 숫자형 변환 (에러 방지용)
    for col in ['강수량', '유량', '강수량_SMA', '유량_SMA']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['날짜'] = df['날짜'].astype(str).str[:8]
    return df.dropna(subset=['강수량', '유량']).sort_values('날짜').reset_index(drop=True)

df = load_processed_data()

# [3. 메인 분석 화면]
if df is not None and len(df) >= 31: # 30일 데이터 확보 확인
    scaler = MinMaxScaler()
    # 모델 학습 데이터 구조에 맞춰 가공된 데이터를 사용합니다.
    scaled = scaler.fit_transform(df[['강수량_SMA', '유량_SMA']].values)

    st.sidebar.header("📅 분석 시점 선택")
    date_list = df['날짜'].tolist()[30:] 
    selected_t = st.sidebar.selectbox("기준일(T) 선택", date_list, index=len(date_list)-1)
    
    pos = df[df['날짜'] == selected_t].index[0]
    
    # [4. 예측 실행]
    input_seq = scaled[pos-29 : pos+1].reshape(1, 30, 2)
    preds_scaled = model.predict(input_seq, verbose=0)[0]
    
    def to_real(s_val):
        dummy = np.zeros((1, 2)); dummy[0, 1] = s_val
        return scaler.inverse_transform(dummy)[0, 1]
    
    preds_real = [to_real(p) for p in preds_scaled]

    # [5. 결과 출력]
    st.subheader(f"📌 {selected_t} 기준 향후 3일 예측 (30일 분석)")
    c1, c2, c3 = st.columns(3)
    for i in range(3):
        with [c1, c2, c3][i]:
            st.metric(f"T+{i+1}일", f"{preds_real[i]:.2f} m³/s")
            if pos + i + 1 < len(df):
                actual = df['유량'].iloc[pos + i + 1]
                st.write(f"실제값: {actual:.2f} | 오차: {abs(preds_real[i]-actual)/actual*100:.1f}%")

    st.divider()
    st.subheader("📈 데이터 흐름 (3일 이동 평균 적용)")
    st.line_chart(df.iloc[max(0, pos-40) : pos+1].set_index('날짜')[['유량', '유량_SMA']])

else:
    st.warning("데이터가 부족합니다 (최소 31일치 필요). get_data.py를 실행해 넉넉한 기간의 데이터를 업로드해 주세요.")
