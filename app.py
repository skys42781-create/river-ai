import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="한강 유량 정밀 예측", layout="wide")
st.title("🌊 맞춤형 유량 예측 및 오차 분석")

@st.cache_resource
def get_model():
    return load_model('river_model_final.h5', compile=False)

model = get_model()

@st.cache_data
def load_data():
    csv_url = "https://raw.githubusercontent.com/skys42781-create/river-ai/main/latest_river_data.csv"
    df = pd.read_csv(csv_url)
    
    # [핵심 수정] 어떤 이름으로 되어있든 우리 방식으로 통일 (유연한 매핑)
    # 영어 약자(rf, fw)와 기존 한글(강수량(mm)) 모두 대응
    name_map = {
        'rf': '강수량', 'rf(mm)': '강수량', '강수량(mm)': '강수량',
        'fw': '유량', 'fw(m3/s)': '유량', '유량(m3/s)': '유량',
        'ymdhm': '날짜', '년월일(yyyyMMdd)': '날짜'
    }
    df = df.rename(columns=name_map)
    
    # 필요한 컬럼만 추출하고 수치형으로 변환
    cols = ['날짜', '강수량', '유량']
    df = df[cols]
    df[['강수량', '유량']] = df[['강수량', '유량']].apply(pd.to_numeric, errors='coerce')
    
    return df.dropna().query("강수량 >= 0 and 유량 >= 0").sort_values('날짜')

df = load_data()

# 사이드바 설정
st.sidebar.header("⚙️ 예측 설정")
target_t = st.sidebar.slider("예측 목표일(T) 선택", 1, 3, 1)

if df is not None and len(df) >= 10:
    scaler = MinMaxScaler()
    data_val = df[['강수량', '유량']].values
    scaled = scaler.fit_transform(data_val)
    
    # 오차율 계산 (MAE, MAPE 등)
    # 윈도우 슬라이딩 방식으로 과거 시점들 예측해보기
    test_input, test_real = [], []
    for i in range(len(scaled) - 7 - target_t):
        test_input.append(scaled[i:i+7])
        test_real.append(scaled[i+7+(target_t-1), 1])
    
    test_input = np.array(test_input)
    y_pred_scaled = model.predict(test_input, verbose=0)[:, target_t-1]
    
    def denormalize(val):
        d = np.zeros((len(val), 2))
        d[:, 1] = val
        return scaler.inverse_transform(d)[:, 1]

    y_real = denormalize(np.array(test_real))
    y_pred = denormalize(y_pred_scaled)
    
    # 성능 지표 출력
    mae = mean_absolute_error(y_real, y_pred)
    mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100

    st.subheader(f"📊 T+{target_t} 성능 지표")
    m1, m2 = st.columns(2)
    m1.metric("평균 오차 (MAE)", f"{mae:.2f} m³/s")
    m2.metric("오차율 (MAPE)", f"{mape:.1f} %")

    # 최종 실시간 예측
    st.divider()
    curr_input = scaled[-7:].reshape(1, 7, 2)
    final_pred_scaled = model.predict(curr_input, verbose=0)[0, target_t-1]
    
    dummy = np.zeros((1, 2))
    dummy[0, 1] = final_pred_scaled
    final_pred = scaler.inverse_transform(dummy)[0, 1]
    
    st.subheader(f"📅 실시간 T+{target_t} 예측값")
    st.info(f"예측 유량: **{final_pred:.2f} m³/s**")
    
    # 그래프
    chart_df = pd.DataFrame({'실제값': y_real, '예측값': y_pred})
    st.line_chart(chart_df.tail(20))
