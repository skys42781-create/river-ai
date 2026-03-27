import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# [1. 페이지 설정 및 모델 로드]
st.set_page_config(page_title="한강 수문 정밀 분석 시스템", layout="wide")
st.title("🌊 한강 수문 빅데이터 정밀 분석 및 AI 예측")

LOOK_BACK = 120 

@st.cache_resource
def get_model():
    # 2층 LSTM + 로그 변환 학습 모델 로드
    return load_model('river_model_final.h5', compile=False)

try:
    model = get_model()
except Exception as e:
    st.error(f"❌ 모델 로드 실패: {e}. 깃허브에 최신 모델(.h5)이 있는지 확인하세요.")
    st.stop()

# [2. 데이터 로드 및 로그 변환]
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
    # 유량 데이터가 확정된 2007년 이후 데이터만 추출
    df = df.dropna(subset=['유량']).query("유량 > 0").sort_values('날짜').reset_index(drop=True)
    
    # 학습 환경과 동일한 로그 변환 적용
    df['rf_log'] = np.log1p(df['강수량'])
    df['fw_log'] = np.log1p(df['유량'])
    return df

df = load_river_data()

# [3. 메인 분석 로직]
if len(df) >= LOOK_BACK + 1:
    date_options = df['날짜'].tolist()[LOOK_BACK:]
    
    # 사이드바 설정
    st.sidebar.header("🕒 분석 시점 및 위험도")
    live_mode = st.sidebar.toggle("최신 확정 데이터 자동 추적", value=True)
    selected_t = date_options[-1] if live_mode else st.sidebar.selectbox("기준일(T) 선택", date_options, index=len(date_options)-1)
    
    pos = df[df['날짜'] == selected_t].index[0]
    
    # 위험도 판별 (간단 로직)
    curr_fw = df['유량'].iloc[pos]
    if curr_fw > df['유량'].quantile(0.95):
        st.sidebar.error("⚠️ 현재 위험도: [심각] 홍수 주의")
    elif curr_fw > df['유량'].quantile(0.80):
        st.sidebar.warning("🟡 현재 위험도: [주의] 유량 급증")
    else:
        st.sidebar.success("🟢 현재 위험도: [정상] 안정적")

    # --- 데이터 준비 및 예측 ---
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['rf_log', 'fw_log']].values)
    
    def get_real_prediction(seq):
        pred_scaled = model.predict(seq, verbose=0)[0]
        real_list = []
        for p in pred_scaled:
            dummy = np.zeros((1, 2))
            dummy[0, 1] = p
            log_val = scaler.inverse_transform(dummy)[0, 1]
            real_list.append(np.expm1(log_val))
        return real_list

    # 현재 시점(T) 기반 예측
    input_seq = scaled[pos-(LOOK_BACK-1) : pos+1].reshape(1, LOOK_BACK, 2)
    preds_real = get_real_prediction(input_seq)

    # --- 구역 1: 기본 수문 Metrics ---
    st.subheader(f"📊 {selected_t} 기준 수문 실시간 대시보드")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("현재 유량", f"{curr_fw:.2f} m³/s", f"{curr_fw - df['유량'].iloc[pos-1]:+.2f}")
    m2.metric("현재 수위", f"{df['수위'].iloc[pos]:.2f} m")
    m3.metric("최근 7일 강우", f"{df['강수량'].iloc[pos-6:pos+1].sum():.1f} mm")
    m4.metric("연간 최고 유량", f"{df['유량'].max():.2f} m³/s")

    # --- 구역 2: [신규] 수문 공학적 정밀 지표 ---
    st.divider()
    st.subheader("🧪 수문 공학적 정밀 진단 지표")
    c1, c2, c3 = st.columns(3)
    
    # 1. 유출 계수 (최근 30일 강우 대비 유량 비율)
    recent_30 = df.iloc[pos-29:pos+1]
    total_rf = recent_30['강수량'].sum()
    total_fw = recent_30['유량'].sum()
    runoff_coeff = min((total_fw / (total_rf * 10 + 0.1)), 1.0) # 간이 보정치 적용
    
    # 2. NSE (모델 신뢰도 지표) 계산
    actual_30, pred_30 = [], []
    for p in range(pos-15, pos): # 계산 속도를 위해 최근 15일 검증
        t_seq = scaled[p-(LOOK_BACK-1) : p+1].reshape(1, LOOK_BACK, 2)
        actual_30.append(df['유량'].iloc[p+1])
        pred_30.append(get_real_prediction(t_seq)[0])
    
    actual_30, pred_30 = np.array(actual_30), np.array(pred_30)
    nse = 1 - (np.sum((actual_30 - pred_30)**2) / np.sum((actual_30 - np.mean(actual_30))**2))
    
    # 3. 기저 유량 (Baseflow) 산출 (연간 하위 10% 유량)
    baseflow = df['유량'].quantile(0.1)

    with c1:
        st.metric("유역 유출 계수", f"{runoff_coeff:.2f}", help="강우가 하천 유량으로 변환되는 효율입니다.")
    with c2:
        st.metric("모델 신뢰도 (NSE)", f"{nse:.2f}", delta="Excellent" if nse > 0.7 else "Normal")
    with c3:
        st.metric("기저 유량 (Baseflow)", f"{baseflow:.2f} m³/s", help="가뭄 시에도 유지되는 최소 유량입니다.")

    # --- 구역 3: T+1~T+3 예측 결과 ---
    st.divider()
    st.subheader("🔮 향후 3일(T+1~T+3) AI 유량 예측")
    p1, p2, p3 = st.columns(3)
    for i in range(3):
        with [p1, p2, p3][i]:
            st.info(f"**T+{i+1}일 예측**")
            st.title(f"{preds_real[i]:.2f}")
            if pos + i + 1 < len(df):
                act = df['유량'].iloc[pos+i+1]
                err = abs(preds_real[i] - act) / act * 100
                st.write(f"실제값: {act:.2f} (오차 {err:.1f}%)")

    # --- 구역 4: 오차 및 트렌드 시각화 ---
    st.divider()
    l_chart, r_chart = st.columns(2)
    with l_chart:
        st.write("**📈 최근 120일 유량 및 강수 흐름**")
        st.line_chart(df.iloc[pos-119:pos+1].set_index('날짜')[['유량', '강수량']])
    with r_chart:
        st.write("**📉 최근 15일간의 오차율 변동 (%)**")
        error_df = pd.DataFrame({'날짜': df['날짜'].iloc[pos-14:pos+1], 
                                 '오차율': [abs(p-a)/a*100 for p, a in zip(pred_30, actual_30)]}).set_index('날짜')
        st.area_chart(error_df, color="#ff4b4b")

else:
    st.error(f"데이터가 부족합니다. (최소 {LOOK_BACK+1}일 필요)")
