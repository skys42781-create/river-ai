import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# [1. 설정 및 모델 로드]
st.set_page_config(page_title="한강 수문 정밀 예측 시스템", layout="wide")
st.title("🌊 120일 로그 변환 모델 기반 유량 예측")

LOOK_BACK = 120 

@st.cache_resource
def get_model():
    # 새로 만든 h5 파일을 로드합니다.
    return load_model('river_model_final.h5', compile=False)

try:
    model = get_model()
except Exception as e:
    st.error(f"❌ 모델 로드 실패: {e}")
    st.stop()

# [2. 데이터 로드 및 로그 변환 적용]
@st.cache_data
def load_river_data():
    csv_url = "https://raw.githubusercontent.com/skys42781-create/river-ai/main/latest_river_data.csv"
    df_raw = pd.read_csv(csv_url)
    df_raw.columns = df_raw.columns.str.strip()
    
    processed = {}
    mapping = {'강수량': ['rf', '강수'], '유량': ['fw', '유량'], '수위': ['wl', '수위'], '날짜': ['ymd', '날짜', '년월일']}
    for target, keys in mapping.items():
        for col in df_raw.columns:
            if any(k in col.lower() for k in keys):
                if target == '날짜': processed[target] = df_raw[col].astype(str).str[:8]
                else: processed[target] = pd.to_numeric(df_raw[col], errors='coerce')
                break
    
    df = pd.DataFrame(processed)
    df = df.dropna(subset=['유량']).query("유량 > 0").sort_values('날짜').reset_index(drop=True)
    
    # [중요] 학습 환경과 동일하게 입력 데이터에 로그 변환 적용
    df['rf_log'] = np.log1p(df['강수량'])
    df['fw_log'] = np.log1p(df['유량'])
    
    return df

df = load_river_data()

# [3. 예측 및 시각화]
if len(df) >= LOOK_BACK + 1:
    date_options = df['날짜'].tolist()[LOOK_BACK:]
    selected_t = st.sidebar.selectbox("기준일(T) 선택", date_options, index=len(date_options)-1)
    pos = df[df['날짜'] == selected_t].index[0]

    # 스케일러 설정 (학습 시 사용한 순서: rf_log, fw_log)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['rf_log', 'fw_log']].values)
    input_seq = scaled[pos-(LOOK_BACK-1) : pos+1].reshape(1, LOOK_BACK, 2)

    # 예측 및 역변환 함수
    def get_real_prediction(seq):
        pred_scaled = model.predict(seq, verbose=0)[0]
        real_list = []
        for p in pred_scaled:
            # 1. 스케일링 복원 (로그 값 상태)
            dummy = np.zeros((1, 2))
            dummy[0, 1] = p
            log_val = scaler.inverse_transform(dummy)[0, 1]
            # 2. 지수 변환 (실제 유량 숫자로 복원)
            real_val = np.expm1(log_val)
            real_list.append(real_val)
        return real_list

    preds = get_real_prediction(input_seq)

    # --- 화면 구성 ---
    st.subheader(f"📊 {selected_t} 예측 결과 (T+1 ~ T+3)")
    cols = st.columns(3)
    for i in range(3):
        with cols[i]:
            st.info(f"T+{i+1}일 예측")
            st.title(f"{preds[i]:.2f}")
            if pos + i + 1 < len(df):
                actual = df['유량'].iloc[pos+i+1]
                error = abs(preds[i] - actual) / actual * 100
                st.write(f"실제: {actual:.2f} (오차 {error:.1f}%)")

    # --- 오차 추이 그래프 ---
    st.divider()
    st.subheader("📉 최근 30일간의 오차율 변동")
    error_log = []
    for p in range(pos-30, pos):
        if p < LOOK_BACK: continue
        t_seq = scaled[p-(LOOK_BACK-1) : p+1].reshape(1, LOOK_BACK, 2)
        t_pred = get_real_prediction(t_seq)[0]
        t_actual = df['유량'].iloc[p+1]
        error_log.append({'날짜': df['날짜'].iloc[p+1], '오차율(%)': (abs(t_pred-t_actual)/t_actual)*100})
    
    if error_log:
        st.area_chart(pd.DataFrame(error_log).set_index('날짜'))

else:
    st.error("데이터 부족")
