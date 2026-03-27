import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# [1. 설정 및 모델 로드]
st.set_page_config(page_title="한강 수문 정밀 검증 시스템", layout="wide")
st.title("📅 365일 빅데이터 기반 유량 예측 및 오차 분석")

@st.cache_resource
def get_model():
    return load_model('river_model_final.h5', compile=False)

try:
    model = get_model()
except Exception as e:
    st.error(f"❌ 모델 로드 실패: {e}")
    st.stop()

# [2. 데이터 로드 및 정제]
@st.cache_data
def load_master_data():
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
    # 유량 데이터가 있는 2007년 이후 데이터만 사용
    df = df.dropna(subset=['유량']).query("유량 > 0").sort_values('날짜').reset_index(drop=True)
    return df

df = load_master_data()

# [3. 분석 및 예측 로직]
if len(df) >= 366:
    date_options = df['날짜'].tolist()[365:]
    st.sidebar.header("🕒 분석 시점 설정")
    live_mode = st.sidebar.toggle("유량 데이터가 있는 최신일 자동 추적", value=True)
    selected_t = date_options[-1] if live_mode else st.sidebar.selectbox("날짜 선택", date_options, index=len(date_options)-1)
    
    pos = df[df['날짜'] == selected_t].index[0]
    
    # [4. AI 예측 수행]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['강수량', '유량']].values)
    input_seq = scaled[pos-364 : pos+1].reshape(1, 365, 2)
    
    preds_real = []
    try:
        preds_scaled = model.predict(input_seq, verbose=0)[0]
        def to_real(s_val):
            dummy = np.zeros((1, 2)); dummy[0, 1] = s_val
            return scaler.inverse_transform(dummy)[0, 1]
        preds_real = [to_real(p) for p in preds_scaled]
    except Exception as e:
        st.error(f"예측 오류: {e}")

    # --- 구역 1: 주요 요약 Metrics ---
    st.subheader(f"📊 {selected_t} (T) 기준 수문 분석")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("현재 유량", f"{df['유량'].iloc[pos]:.2f} m³/s")
    m2.metric("현재 수위", f"{df['수위'].iloc[pos]:.2f} m")
    m3.metric("최근 30일 강우", f"{df['강수량'].iloc[pos-29:pos+1].sum():.1f} mm")
    
    # --- 구역 2: T+1 ~ T+3 예측 결과 ---
    st.divider()
    st.subheader("🔮 향후 3일(T+1~T+3) AI 유량 예측")
    c1, c2, c3 = st.columns(3)
    for i in range(3):
        with [c1, c2, c3][i]:
            val = preds_real[i] if i < len(preds_real) else 0
            st.info(f"**T+{i+1}일 예측**")
            st.title(f"{val:.2f}")
            # 검증 데이터가 존재하면 오차 계산
            if pos + i + 1 < len(df):
                actual = df['유량'].iloc[pos+i+1]
                error_pct = abs(val - actual) / actual * 100
                st.write(f"실제값: {actual:.2f} (오차: {error_pct:.1f}%)")

    # --- 구역 3: [신규] 오차 변동 그래프 (사후 검증 섹션) ---
    st.divider()
    st.subheader("📉 모델 성능 검증: 최근 30일간의 오차 변동")
    
    # 최근 30일 동안 각 시점(T)에서 T+1일 예측을 수행했을 때의 오차를 계산
    error_data = []
    analysis_range = range(pos-30, pos) # 선택한 날짜 기준 과거 30일
    
    for p in analysis_range:
        if p < 365: continue
        # 각 날짜별로 365일 데이터를 모델에 넣고 T+1 예측값 추출
        test_input = scaled[p-364 : p+1].reshape(1, 365, 2)
        test_pred_scaled = model.predict(test_input, verbose=0)[0][0] # T+1 예측값
        
        # 실제값으로 복원
        dummy = np.zeros((1, 2)); dummy[0, 1] = test_pred_scaled
        pred_val = scaler.inverse_transform(dummy)[0, 1]
        
        # 실제 관측값과 비교
        actual_val = df['유량'].iloc[p+1]
        abs_error = abs(pred_val - actual_val)
        error_data.append({
            '날짜': df['날짜'].iloc[p+1],
            '예측오차(m3/s)': abs_error,
            '오차율(%)': (abs_error / actual_val) * 100
        })

    error_df = pd.DataFrame(error_data).set_index('날짜')
    
    col_err1, col_err2 = st.columns(2)
    with col_err1:
        st.write("**절대 오차 변동 (m³/s)**")
        st.line_chart(error_df['예측오차(m3/s)'])
    with col_err2:
        st.write("**상대 오차율 변동 (%)**")
        st.area_chart(error_df['오차율(%)'])

    # --- 구역 4: 연간 유량 트렌드 ---
    st.divider()
    st.subheader("📈 지난 1년간의 유량 변동 추이")
    st.line_chart(df.iloc[pos-364 : pos+1].set_index('날짜')['유량'])

else:
    st.error("분석을 위해 최소 366일의 유량 데이터가 필요합니다.")
