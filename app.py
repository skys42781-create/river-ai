import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# [1. 페이지 설정 및 모델 로드]
st.set_page_config(page_title="포천천 화룡교 수문 분석", layout="wide")
st.title("🌊 포천천(화룡교) 빅데이터 기반 AI 유량 예측 시스템")

LOOK_BACK = 120  # 4개월(120일) 분석 창

@st.cache_resource
def get_model():
    # 2층 LSTM + 로그 변환으로 학습된 모델 파일 (.h5)
    return load_model('river_model_final.h5', compile=False)

try:
    model = get_model()
except Exception as e:
    st.error(f"❌ 모델 로드 실패: {e}. 'river_model_final.h5' 파일이 경로에 있는지 확인하세요.")
    st.stop()

# [2. 데이터 로드 및 전처리]
@st.cache_data
def load_data():
    # 근우님의 깃허브 원본 데이터 경로
    csv_url = "https://raw.githubusercontent.com/skys42781-create/river-ai/main/latest_river_data.csv"
    df_raw = pd.read_csv(csv_url)
    df_raw.columns = df_raw.columns.str.strip()
    
    processed = {}
    # 화룡교 관측소 컬럼 매핑
    mapping = {'강수량': ['rf', '강수'], '유량': ['fw', '유량'], '수위': ['wl', '수위'], '날짜': ['ymd', '날짜', '년월일']}
    for target, keys in mapping.items():
        for col in df_raw.columns:
            if any(k in col.lower() for k in keys):
                if target == '날짜': processed[target] = df_raw[col].astype(str).str[:8]
                else: processed[target] = pd.to_numeric(df_raw[col], errors='coerce')
                break
    
    df = pd.DataFrame(processed)
    # 유량 확정 데이터가 존재하는 시점부터 추출
    df = df.dropna(subset=['유량']).query("유량 > 0").sort_values('날짜').reset_index(drop=True)
    
    # 학습 시와 동일하게 로그 변환 적용 (중요!)
    df['rf_log'] = np.log1p(df['강수량'])
    df['fw_log'] = np.log1p(df['유량'])
    return df

df = load_data()

# [3. 메인 분석 및 예측 로직]
if len(df) >= LOOK_BACK + 1:
    date_options = df['날짜'].tolist()[LOOK_BACK:]
    
    st.sidebar.header("🕒 화룡교 분석 설정")
    live_mode = st.sidebar.toggle("최신 확정 데이터 자동 추적", value=True)
    selected_t = date_options[-1] if live_mode else st.sidebar.selectbox("기준일(T) 선택", date_options, index=len(date_options)-1)
    
    pos = df[df['날짜'] == selected_t].index[0]
    
    # 위험도 판별 (화룡교 유량 통계 기준)
    curr_fw = df['유량'].iloc[pos]
    if curr_fw > df['유량'].quantile(0.95):
        st.sidebar.error("⚠️ [심각] 화룡교 홍수 주의 단계")
    elif curr_fw > df['유량'].quantile(0.80):
        st.sidebar.warning("🟡 [주의] 유량 급증 단계")
    else:
        st.sidebar.success("🟢 [정상] 하천 수위 안정")

    # --- 스케일링 및 예측 함수 ---
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['rf_log', 'fw_log']].values)
    
    def predict_flow(seq):
        pred_scaled = model.predict(seq, verbose=0)[0]
        results = []
        for p in pred_scaled:
            dummy = np.zeros((1, 2))
            dummy[0, 1] = p
            log_val = scaler.inverse_transform(dummy)[0, 1]
            results.append(np.expm1(log_val)) # 로그 역변환
        return results

    # 현재 예측
    input_seq = scaled[pos-(LOOK_BACK-1) : pos+1].reshape(1, LOOK_BACK, 2)
    preds = predict_flow(input_seq)

    # --- 구역 1: 기본 현황 Metrics ---
    st.subheader(f"📊 {selected_t} 화룡교 지점 수문 현황")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("현재 유량", f"{curr_fw:.2f} m³/s", f"{curr_fw - df['유량'].iloc[pos-1]:+.2f}")
    m2.metric("현재 수위", f"{df['수위'].iloc[pos]:.2f} m")
    m3.metric("최근 7일 강우", f"{df['강수량'].iloc[pos-6:pos+1].sum():.1f} mm")
    m4.metric("분석 창(Window)", f"{LOOK_BACK}일")

    # --- 구역 2: 수문 공학적 정밀 지표 ---
    st.divider()
    st.subheader("🧪 포천천 유역 정밀 진단 지표")
    c1, c2, c3 = st.columns(3)
    
    # NSE 및 오차 데이터 생성 (최근 15일)
    actual_15, pred_15 = [], []
    for p in range(pos-15, pos):
        t_seq = scaled[p-(LOOK_BACK-1) : p+1].reshape(1, LOOK_BACK, 2)
        actual_15.append(df['유량'].iloc[p+1])
        pred_15.append(predict_flow(t_seq)[0])
    
    actual_15, pred_15 = np.array(actual_15), np.array(pred_15)
    nse = 1 - (np.sum((actual_15 - pred_15)**2) / np.sum((actual_15 - np.mean(actual_15))**2))
    runoff_ratio = (df['유량'].iloc[pos-29:pos+1].sum() / (df['강수량'].iloc[pos-29:pos+1].sum() * 10 + 0.1))

    with c1:
        st.metric("유역 유출 계수", f"{min(runoff_ratio, 1.0):.2f}", help="강우 대비 유량 변환 효율")
    with c2:
        st.metric("모델 신뢰도 (NSE)", f"{nse:.2f}", delta="우수" if nse > 0.6 else "보통")
    with c3:
        st.metric("기저 유량", f"{df['유량'].quantile(0.1):.2f} m³/s", help="하천 유지 최소 유량")

    # --- 구역 3: 향후 3일 AI 예측 ---
    st.divider()
    st.subheader("🔮 향후 3일(T+1~T+3) AI 유량 예측")
    p1, p2, p3 = st.columns(3)
    for i in range(3):
        with [p1, p2, p3][i]:
            st.info(f"**T+{i+1}일 예측**")
            st.title(f"{preds[i]:.2f}")
            if pos + i + 1 < len(df):
                act = df['유량'].iloc[pos+i+1]
                st.write(f"실제값: {act:.2f} (오차 {abs(preds[i]-act)/act*100:.1f}%)")

    # --- 구역 4: 상세 분석 차트 ---
    st.divider()
    st.subheader("🔍 데이터 입체 분석 (상관관계 및 오차 분포)")
    col_l, col_r = st.columns(2)
    with col_l:
        st.write("**🌧️ 강수량-유량 상관관계 (최근 120일)**")
        st.scatter_chart(df.iloc[pos-119:pos+1], x='강수량', y='유량')
    with col_r:
        st.write("**📉 최근 15일간의 오차율 변동 (%)**")
        err_df = pd.DataFrame({'날짜': df['날짜'].iloc[pos-14:pos+1], 
                               '오차율': [abs(p-a)/a*100 for p, a in zip(pred_15, actual_15)]}).set_index('날짜')
        st.area_chart(err_df, color="#ff4b4b")

    st.divider()
    st.write("**📈 포천천 화룡교 유량 변동 추이 (최근 120일)**")
    st.line_chart(df.iloc[pos-119:pos+1].set_index('날짜')['유량'])

else:
    st.error("분석을 위한 데이터가 충분하지 않습니다.")
