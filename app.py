import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# [1. 기본 설정]
st.set_page_config(page_title="한강 유량 시점별 분석", layout="wide")
st.title("🌊 특정 시점(T) 기준 향후 3일 예측 및 검증")

@st.cache_resource
def get_model():
    return load_model('river_model_final.h5', compile=False)

model = get_model()

# [2. 데이터 로드: 이름 충돌 방지 로직 강화]
@st.cache_data
def load_data():
    csv_url = "https://raw.githubusercontent.com/skys42781-create/river-ai/main/latest_river_data.csv"
    df = pd.read_csv(csv_url)
    
    # 원본 파일의 컬럼명을 유연하게 한글로 통일
    name_map = {
        'rf': '강수량', 'rf(mm)': '강수량', '강수량(mm)': '강수량',
        'fw': '유량', 'fw(m3/s)': '유량', '유량(m3/s)': '유량',
        'ymdhm': '날짜', '년월일(yyyyMMdd)': '날짜'
    }
    df = df.rename(columns=name_map)
    
    # 필요한 데이터만 추출
    df = df[['날짜', '강수량', '유량']]
    df['강수량'] = pd.to_numeric(df['강수량'], errors='coerce')
    df['유량'] = pd.to_numeric(df['유량'], errors='coerce')
    
    # 날짜 형식 정리 (앞 8자리)
    df['날짜'] = df['날짜'].astype(str).str[:8]
    return df.dropna().query("강수량 >= 0 and 유량 >= 0").sort_values('날짜').reset_index(drop=True)

df = load_data()

# [3. 메인 분석 로직]
if df is not None and len(df) >= 10:
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['강수량', '유량']].values)
    
    # 사이드바에서 기준 날짜(T) 선택
    st.sidebar.header("📅 분석 시점 설정")
    date_list = df['날짜'].tolist()[7:] # 최소 7일 데이터 확보된 날짜부터
    selected_t = st.sidebar.selectbox("기준 날짜(T) 선택", date_list, index=len(date_list)-1)
    
    # 선택된 날짜의 위치 찾기
    pos = df[df['날짜'] == selected_t].index[0]
    
    # T 시점의 7일 데이터를 입력으로 사용
    input_seq = scaled[pos-6 : pos+1].reshape(1, 7, 2)
    preds_scaled = model.predict(input_seq, verbose=0)[0]
    
    def to_real(s_val):
        dummy = np.zeros((1, 2)); dummy[0, 1] = s_val
        return scaler.inverse_transform(dummy)[0, 1]
    
    preds_real = [to_real(p) for p in preds_scaled]

    # [4. 결과 출력: T+1, T+2, T+3]
    st.subheader(f"📌 {selected_t} (T) 시점 기준 예측 결과")
    cols = st.columns(3)
    
    for i in range(3):
        with cols[i]:
            st.metric(f"T+{i+1}일 예측", f"{preds_real[i]:.2f} m³/s")
            
            # 사후 검증 (실제값이 데이터에 존재하는 경우)
            if pos + i + 1 < len(df):
                actual = df['유량'].iloc[pos + i + 1]
                diff = preds_real[i] - actual
                error_pct = (abs(diff) / actual) * 100
                st.write(f"**실제 관측치:** {actual:.2f}")
                st.write(f"**오차율:** {error_pct:.1f}%")
            else:
                st.write("*(미래 시점으로 실제값 없음)*")

    # [5. 데이터 테이블 확인]
    st.divider()
    st.subheader("📊 데이터 무결성 확인 (선택 시점 주변)")
    st.table(df.iloc[max(0, pos-5) : pos+4])

else:
    st.error("데이터를 불러오지 못했거나 양이 부족합니다.")
