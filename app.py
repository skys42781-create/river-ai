import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# [1. 설정]
st.set_page_config(page_title="한강 유량 정밀 예측", layout="wide")
st.title("🌊 맞춤형 유량 예측 및 오차 분석 시스템")

@st.cache_resource
def get_model():
    return load_model('river_model_final.h5', compile=False)

model = get_model()

# [2. 데이터 로드]
@st.cache_data
def load_data():
    csv_url = "https://raw.githubusercontent.com/skys42781-create/river-ai/main/latest_river_data.csv"
    df = pd.read_csv(csv_url)
    rename_rule = {'rf': '강수량', 'fw': '유량', 'ymdhm': '날짜', '년월일(yyyyMMdd)': '날짜'}
    df = df.rename(columns=rename_rule)
    df[['강수량', '유량']] = df[['강수량', '유량']].apply(pd.to_numeric, errors='coerce')
    return df.dropna().query("강수량 >= 0 and 유량 >= 0").sort_values('날짜')

df = load_data()

# [3. 컨트롤 패널: T값 설정]
st.sidebar.header("⚙️ 예측 설정")
target_t = st.sidebar.slider("예측 목표일(T) 선택", min_value=1, max_value=3, value=1, help="오늘(T=0) 기준으로 며칠 뒤를 볼지 정합니다.")

if df is not None and len(df) >= 10:
    # 데이터 전처리
    scaler = MinMaxScaler()
    data_val = df[['강수량', '유량']].values
    scaled = scaler.fit_transform(data_val)
    
    # [4. 실시간 오차율(Performance) 계산]
    # 최근 데이터를 사용해 과거 예측이 얼마나 정확했는지 검증
    # (주의: 실제 현장에서는 과거 데이터를 밀어넣어 예측값과 실제값의 차이를 구함)
    test_input = []
    test_real = []
    for i in range(len(scaled) - 7 - target_t):
        test_input.append(scaled[i:i+7])
        test_real.append(scaled[i+7+(target_t-1), 1]) # 유량(1번 인덱스) 타겟값
    
    test_input = np.array(test_input)
    y_pred_scaled = model.predict(test_input, verbose=0)[:, target_t-1]
    
    # 역정규화 (오차 계산용)
    def denormalize(val):
        d = np.zeros((len(val), 2))
        d[:, 1] = val
        return scaler.inverse_transform(d)[:, 1]

    y_real_final = denormalize(np.array(test_real))
    y_pred_final = denormalize(y_pred_scaled)
    
    # 지표 계산
    mae = mean_absolute_error(y_real_final, y_pred_final)
    rmse = np.sqrt(mean_squared_error(y_real_final, y_pred_final))
    mape = np.mean(np.abs((y_real_final - y_pred_final) / y_real_final)) * 100

    # [5. 대시보드 출력]
    st.subheader(f"📊 T+{target_t}일 뒤 예측 성능 (최근 데이터 기준)")
    m1, m2, m3 = st.columns(3)
    m1.metric("평균 절대 오차 (MAE)", f"{mae:.2f} m³/s")
    m2.metric("제곱근 평균 오차 (RMSE)", f"{rmse:.2f} m³/s")
    m3.metric("평균 오차율 (MAPE)", f"{mape:.1f} %")

    # [6. 최종 예측 (현재 시점)]
    st.divider()
    curr_input = scaled[-7:].reshape(1, 7, 2)
    final_pred_scaled = model.predict(curr_input, verbose=0)[0, target_t-1]
    
    # 단일 값 역정규화
    dummy = np.zeros((1, 2))
    dummy[0, 1] = final_pred_scaled
    final_pred = scaler.inverse_transform(dummy)[0, 1]
    
    st.subheader(f"📅 실시간 예측: T+{target_t}일 유량")
    st.markdown(f"### 예상 유량: <span style='color:blue'>{final_pred:.2f} m³/s</span>", unsafe_allow_html=True)
    
    # [7. 시각화] 실제 vs 예측 비교
    chart_df = pd.DataFrame({
        '실제 관측치': y_real_final,
        '모델 예측치': y_pred_final
    })
    st.line_chart(chart_df.tail(20))
