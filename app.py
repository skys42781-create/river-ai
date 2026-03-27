import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# 한글 깨짐 방지 설정
plt.rc('font', family='NanumBarunGothic') 
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="유량 예측 시스템", layout="wide")
st.title("🌊 실시간 하천 유량 예측 및 성능 검증")

# 모델 불러오기 (에러 방지를 위해 compile=False 필수)
@st.cache_resource
def get_model():
    try:
        # 이 부분이 핵심입니다!
        return load_model('river_model_final.h5', compile=False)
    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {e}")
        return None

model = get_model()

# 사이드바 설정
st.sidebar.header("📁 데이터 업데이트")
rain_file = st.sidebar.file_uploader("최신 강수량 엑셀 업로드", type=['xlsx'])
flow_file = st.sidebar.file_uploader("최신 유량 엑셀 업로드", type=['xlsx'])

if rain_file and flow_file and model:
    # 1. 데이터 로드 및 무결성 검토
    r_df = pd.read_excel(rain_file)
    f_df = pd.read_excel(flow_file)
    df = pd.merge(r_df, f_df, on='년월일(yyyyMMdd)').dropna()
    df = df[(df['강수량(mm)'] >= 0) & (df['유량(m3/s)'] >= 0)]
    
    st.success(f"✅ 데이터 반영 완료: {df['년월일(yyyyMMdd)'].iloc[-1]}까지")

    # 2. 데이터 전처리
    data = df[['강수량(mm)', '유량(m3/s)']].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # 3. 미래 3일 예측 (마지막 7일 데이터 기반)
    recent_input = scaled_data[-7:].reshape(1, 7, 2)
    pred_scaled = model.predict(recent_input)[0]
    
    # 역정규화 (수치 복원 함수)
    def to_real(val):
        dummy = np.zeros((1, 2))
        dummy[0, 1] = val
        return scaler.inverse_transform(dummy)[0, 1]

    # 4. 결과 출력 (정확한 수치 표시)
    st.subheader("📅 미래 3일 유량 예측 결과")
    c1, c2, c3 = st.columns(3)
    c1.metric("내일(T+1) 예측값", f"{to_real(pred_scaled[0]):.2f} m³/s")
    c2.metric("모레(T+2) 예측값", f"{to_real(pred_scaled[1]):.2f} m³/s")
    c3.metric("글피(T+3) 예측값", f"{to_real(pred_scaled[2]):.2f} m³/s")

    # 5. 오차율 및 데이터 확인
    st.divider()
    st.subheader("📊 데이터 무결성 검토 및 수치 확인")
    st.write("최근 10일간의 관측 자료입니다.")
    st.dataframe(df.tail(10))

else:
    st.info("왼쪽 사이드바에서 최신 강수량 및 유량 엑셀 파일을 업로드해 주세요.")
