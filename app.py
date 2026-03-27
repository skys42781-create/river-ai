import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# 페이지 설정
st.set_page_config(page_title="실시간 유량 예측 시스템", layout="wide")
st.title("🌊 실시간 하천 유량 예측 및 성능 검증")

# 1. 모델 및 데이터 로드
@st.cache_resource
def get_model():
    return load_model('river_model_final.h5', compile=False)

model = get_model()

st.sidebar.header("📁 데이터 업데이트")
rain_file = st.sidebar.file_uploader("최신 강수량 엑셀 업로드", type=['xlsx'])
flow_file = st.sidebar.file_uploader("최신 유량 엑셀 업로드", type=['xlsx'])

if rain_file and flow_file:
    # 데이터 처리 및 무결성 검사
    r_df = pd.read_excel(rain_file)
    f_df = pd.read_excel(flow_file)
    df = pd.merge(r_df, f_df, on='년월일(yyyyMMdd)').dropna()
    df = df[(df['강수량(mm)'] >= 0) & (df['유량(m3/s)'] >= 0)] # 무결성 검토
    
    st.success(f"✅ 데이터 로드 완료: {df['년월일(yyyyMMdd)'].iloc[-1]}까지 반영됨")

    # 데이터 전처리
    data = df[['강수량(mm)', '유량(m3/s)']].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # 최근 7일 데이터로 예측 준비
    recent_input = scaled_data[-7:].reshape(1, 7, 2)
    pred_scaled = model.predict(recent_input)
    
    # 역정규화 (실제 수치 변환)
    def to_real(val):
        dummy = np.zeros((1, 2))
        dummy[0, 1] = val
        return scaler.inverse_transform(dummy)[0, 1]

    # 결과값 생성
    p1, p2, p3 = pred_scaled[0]
    real_p1, real_p2, real_p3 = to_real(p1), to_real(p2), to_real(p3)

    # 화면 구성 (메트릭)
    col1, col2, col3 = st.columns(3)
    col1.metric("T+1일(내일) 예측", f"{real_p1:.2f} m³/s")
    col2.metric("T+2일(모레) 예측", f"{real_p2:.2f} m³/s")
    col3.metric("T+3일(글피) 예측", f"{real_p3:.2f} m³/s")

    # 과거 오차율 확인 (최근 10일)
    st.subheader("📊 최근 예측 정확도 및 오차율 검토")
    # (과거 예측 로직 추가... 중략)
    # 실제값과 예측값 비교 표 출력
    st.dataframe(df.tail(10)) # 예시로 마지막 데이터 표시

else:
    st.info("왼쪽 사이드바에서 최신 강수량 및 유량 엑셀 파일을 업로드해 주세요.")
