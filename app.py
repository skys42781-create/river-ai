import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# [1. 설정 및 모델 로드]
st.set_page_config(page_title="한강 유량 시점별 예측", layout="wide")
st.title("🌊 특정 시점(T) 기준 향후 3일 예측 분석")

@st.cache_resource
def get_model():
    return load_model('river_model_final.h5', compile=False)

model = get_model()

# [2. 데이터 로드 및 전처리]
@st.cache_data
def load_data():
    csv_url = "https://raw.githubusercontent.com/skys42781-create/river-ai/main/latest_river_data.csv"
    df = pd.read_csv(csv_url)
    name_map = {'rf': '강수량', 'fw': '유량', 'ymdhm': '날짜', '년월일(yyyyMMdd)': '날짜'}
    df = df.rename(columns=name_map)
    df[['강수량', '유량']] = df[['강수량', '유량']].apply(pd.to_numeric, errors='coerce')
    df = df.dropna().query("강수량 >= 0 and 유량 >= 0")
    df['날짜'] = df['날짜'].astype(str).str[:8]
    return df.sort_values('날짜')

df = load_data()

if df is not None and len(df) >= 10:
    # 스케일러 설정
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['강수량', '유량']].values)
    
    # [3. 사이드바: 기준 날짜(T) 선택]
    st.sidebar.header("📅 분석 시점 설정")
    # 최소 7일의 데이터가 필요하므로 선택 가능한 날짜 제한
    date_list = df['날짜'].tolist()[7:] 
    selected_t_date = st.sidebar.selectbox("기준 날짜(T)를 선택하세요", date_list, index=len(date_list)-1)
    
    # 선택한 날짜의 인덱스 찾기
    t_idx = df[df['날짜'] == selected_t_date].index[0]
    # 실제 위치(0부터 시작하는 순서) 계산
    pos = df.index.get_loc(t_idx)
    
    # [4. 예측 실행: 선택한 T시점의 7일 데이터를 입력]
    input_data = scaled[pos-6 : pos+1].reshape(1, 7, 2)
    preds_scaled = model.predict(input_data, verbose=0)[0]
    
    def to_real(s_val):
        dummy = np.zeros((1, 2)); dummy[0, 1] = s_val
        return scaler.inverse_transform(dummy)[0, 1]
    
    preds_real = [to_real(p) for p in preds_scaled]

    # [5. 결과 대시보드]
    st.subheader(f"📌 {selected_t_date} (T) 시점 기준 예측 결과")
    
    c1, c2, c3 = st.columns(3)
    days = ["T+1일", "T+2일", "T+3일"]
    
    for i in range(3):
        with [c1, c2, c3][i]:
            val = preds_real[i]
            st.metric(days[i], f"{val:.2f} m³/s")
            
            # 실제값이 존재하는 경우 오차율 표시 (사후 검증)
            if pos + i + 1 < len(df):
                actual = df['유량'].iloc[pos + i + 1]
                error = abs(actual - val) / actual * 100
                st.write(f"실제값: {actual:.2f}")
                st.write(f"오차율: {error:.1f}%")
            else:
                st.write("실제값: (아직 관측 안 됨)")

    # [6. 시각화]
    st.divider()
    st.subheader("📈 시점별 유량 흐름 (관측 vs 예측)")
    
    # 그래프용 데이터 구성
    plot_df = df.iloc[max(0, pos-10) : pos+4].copy()
    plot_df['유형'] = '실제 관측'
    
    # 예측값 추가를 위한 데이터프레임
    pred_data = []
    for i in range(3):
        pred_data.append({'날짜': f'예측_{i+1}', '유량': preds_real[i], '유형': '모델 예측'})
    
    st.line_chart(df.iloc[pos-10:pos+1].set_index('날짜')['유량'])
    st.info("💡 위 차트는 기준일(T)까지의 실제 유량 변화입니다. 상단 지표에서 예측값을 확인하세요.")

else:
    st.warning("데이터를 불러오는 중입니다...")
