import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MplPolygon
import plotly.graph_objects as go
import google.generativeai as genai

# 1. 페이지 설정
st.set_page_config(page_title="MOSFET AI 시뮬레이터", layout="wide", initial_sidebar_state="expanded")

# UI 최적화 CSS (AI 창 크기 조절 기능 추가)
st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 1rem; max-width: 98%; }
        .header-text { font-size: 20px; font-weight: 700; margin-bottom: 10px; margin-top: 5px; }
        
        /* 사이드바 배경 및 여백 최적화 (스크롤 방지) */
        [data-testid="stSidebar"] { background-color: #f8fafc; border-right: 1px solid #e2e8f0; }
        [data-testid="stSidebarUserContent"] { padding-top: 1.5rem !important; padding-bottom: 1rem !important; }
        
        /* 슬라이더 간격 및 텍스트 박스 간격 대폭 축소 */
        .stSlider { padding-bottom: 0px !important; margin-bottom: -15px !important; }
        .stTextArea { padding-bottom: 0px !important; margin-bottom: -5px !important; }
        hr { margin-top: 10px !important; margin-bottom: 10px !important; }

        /* 우측 AI 컨테이너를 독립된 창처럼 만들고 '크기 조절(resize)' 기능 활성화 */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #f1f5f9;
            border-radius: 12px;
            border: 1px solid #cbd5e1;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            padding: 10px;
            resize: both; /* 마우스 드래그로 가로/세로 크기 조절 허용 */
            overflow: auto; /* 내용이 넘치면 스크롤 생성 */
            min-height: 300px;
            min-width: 250px;
        }
        
        /* 다크모드 시 AI 창 색상 자동 반전 */
        @media (prefers-color-scheme: dark) {
            [data-testid="stVerticalBlockBorderWrapper"] {
                background-color: #1e293b;
                border: 1px solid #334155;
            }
        }
    </style>
""", unsafe_allow_html=True)

# 메인 타이틀
st.markdown("<h2 style='text-align: center; margin-bottom: 20px;'>🔌 MOSFET 물리 시뮬레이터</h2>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. 좌측 사이드바: 제어 및 입력 패널
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### **🎛️ 제어 및 입력 패널**")
    
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    else:
        api_key = st.text_input("Gemini API Key", type="password")
    
    st.markdown("---")
    mos_type = st.selectbox("소자 타입 선택", ["NMOS", "PMOS"])
    
    # 입력 순서 최적화: 문턱 전압 -> 게이트 전압 -> 드레인 전압
    if mos_type == "NMOS":
        v_th = st.slider("문턱 전압 (V_TH) [V]", 0.5, 2.0, 1.0, 0.1)
        v_gs = st.slider("게이트 전압 (V_GS) [V]", 0.0, 5.0, 3.0, 0.1)
        v_ds = st.slider("드레인 전압 (V_DS) [V]", 0.0, 5.0, 1.6, 0.1)
    else:
        v_th = st.slider("문턱 전압 (V_TH) [V]", -2.0, -0.5, -1.0, 0.1)
        v_gs = st.slider("게이트 전압 (V_GS) [V]", -5.0, 0.0, -3.0, 0.1)
        v_ds = st.slider("드레인 전압 (V_DS) [V]", -5.0, 0.0, -1.6, 0.1)

    st.markdown("---")
    
    user_query = st.text_area("질문 입력 구역", "현재 전압 상태를 물리적으로 설명해줘.", height=60, label_visibility="collapsed")
    ask_ai_btn = st.button("AI 실시간 해설 받기", type="primary", use_container_width=True)

# ---------------------------------------------------------
# 3. 물리 계산 로직 (수식 교정 및 Plotly 동기화)
# ---------------------------------------------------------
k_n = 1.0
lambda_mod = 0.02  # 채널 길이 변조 계수 반영

abs_vgs = abs(v_gs)
abs_vds = abs(v_ds)
abs_vth = abs(v_th)
v_ov = abs_vgs - abs_vth
vds_sat = max(v_ov, 0.0)

if abs_vgs < abs_vth:
    op_region, i_d = "차단 영역 (Cutoff)", 0.0
elif abs_vds < v_ov:
    op_region, i_d = "선형 영역 (Linear)", k_n * (v_ov * abs_vds - 0.5 * (abs_vds ** 2))
else:
    op_region, i_d = "포화 영역 (Saturation)", 0.5 * k_n * (v_ov ** 2) * (1 + lambda_mod * (abs_vds - v_ov))

# UI 매핑용 한글명 키벨류
region_kr = "포화 영역 (Saturation)" if "포화" in op_region else "선형 영역 (Linear)" if "선형" in op_region else "차단 영역 (Cutoff)"

# ---------------------------------------------------------
# 4. 메인 대시보드 (3단 레이아웃 구성)
# ---------------------------------------------------------
col1, col2, col3 = st.columns([1, 1.2, 1.2], gap="medium")

# --- [Column 1] 실시간 소자 상태 및 원본 MOSFET 구조 시각화 (유지) ---
with col1:
    st.markdown("<div class='header-text'>📊 실시간 소자 상태</div>", unsafe_allow_html=True)
    
    box_color = "#e6f7ff" if "포화" in op_region else "#f6ffed" if "선형" in op_region else "#fff1f0"
    st.markdown(f"""
    <div style='background-color:{box_color}; padding:10px; border-radius:8px; border:1px solid #ddd; text-align:center;'>
        <div style='font-size:12px; color:#475569; margin-bottom:2px;'>현재 동작 영역</div>
        <span style='font-size:18px; font-weight:700; color:#1e293b;'>{region_kr}</span>
    </div>
    """, unsafe_allow_html=True)
    
    m1, m2 = st.columns(2)
    m1.metric("V_DS,sat" if mos_type == "NMOS" else "|V_DS,sat|", f"{vds_sat:.2f} V")
    m2.metric("드레인 전류 (|I_D|)", f"{i_d:.3f} mA")
    
    st.markdown("<div class='header-text'>🧱 MOSFET 구조 시각화</div>", unsafe_allow_html=True)
    
    # 원본 Matplotlib 구조 시각화 로직 100% 보존
    fig_struct, ax = plt.subplots(figsize=(5, 4.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8.5); ax.axis("off")
    fig_struct.patch.set_facecolor('white')

    sub_color  = "#c8dff0" if mos_type == "NMOS" else "#fce4d6"
    sub_edge   = "#5a8abf" if mos_type == "NMOS" else "#e67e22"
    sub_text   = "p-Substrate" if mos_type == "NMOS" else "n-Substrate"
    sub_tc     = "#2c5f8a" if mos_type == "NMOS" else "#a04000"
    well_text  = "n+" if mos_type == "NMOS" else "p+"
    well_color = "#4caf7d" if mos_type == "NMOS" else "#9b59b6"
    well_edge  = "#2e7d52" if mos_type == "NMOS" else "#8e44ad"
    carrier_color = "#e65100" if mos_type == "NMOS" else "#8e44ad"
    ch_color   = "#66bb6a" if mos_type == "NMOS" else "#d2b4de"
    ch_edge    = "#388e3c" if mos_type == "NMOS" else "#af7ac5"

    sub = patches.FancyBboxPatch((0.3, 0.3), 9.4, 4.8, boxstyle="round,pad=0.1", fc=sub_color, ec=sub_edge, lw=1.5)
    ax.add_patch(sub)
    ax.text(5, 1.0, sub_text, ha="center", va="center", fontsize=10, color=sub_tc, fontstyle='italic')

    src = patches.FancyBboxPatch((0.5, 3.0), 2.3, 2.1, boxstyle="round,pad=0.05", fc=well_color, ec=well_edge, lw=1.5)
    ax.add_patch(src)
    ax.text(1.65, 4.1, "S", ha="center", va="center", fontsize=18, fontweight="bold", color="white")
    ax.text(1.65, 3.35, well_text, ha="center", va="center", fontsize=10, color="#ffffff")

    drn = patches.FancyBboxPatch((7.2, 3.0), 2.3, 2.1, boxstyle="round,pad=0.05", fc=well_color, ec=well_edge, lw=1.5)
    ax.add_patch(drn)
    ax.text(8.35, 4.1, "D", ha="center", va="center", fontsize=18, fontweight="bold", color="white")
    ax.text(8.35, 3.35, well_text, ha="center", va="center", fontsize=10, color="#ffffff")

    sio2 = patches.Rectangle((2.8, 5.1), 4.4, 0.4, fc="#e8e8e8", ec="#aaa", lw=1.0)
    ax.add_patch(sio2)
    ax.text(7.35, 5.3, "SiO2", ha="left", va="center", fontsize=8, color="#9400D3")

    gate = patches.FancyBboxPatch((2.8, 5.5), 4.4, 0.75, boxstyle="round,pad=0.05", fc="#37474f", ec="#1a1a2e", lw=1.5)
    ax.add_patch(gate)
    ax.text(5, 5.88, "Gate (G)", ha="center", va="center", fontsize=10, fontweight="bold", color="white")

    GATE_X_START, GATE_X_END = 2.8, 7.2
    GATE_LEN = GATE_X_END - GATE_X_START
    SIO2_BOTTOM, CH_THICK = 5.1, 0.5

    if "포화" in op_region:
        ratio = float(np.clip(vds_sat / max(abs_vds, 0.01), 0.15, 0.85))
        po_x = GATE_X_START + GATE_LEN * ratio if mos_type == "NMOS" else GATE_X_END - GATE_LEN * ratio
        if mos_type == "NMOS":
            tri_pts = np.array([[GATE_X_START, SIO2_BOTTOM], [po_x, SIO2_BOTTOM], [GATE_X_START, SIO2_BOTTOM - CH_THICK]])
            dep_rect = patches.Rectangle((po_x, SIO2_BOTTOM - CH_THICK), GATE_X_END - po_x, CH_THICK, fc="#dce8f5", ec="#5a8abf", linestyle='--', alpha=0.6)
            arr_start, arr_end = GATE_X_START + 0.2, po_x - 0.15
        else:
            tri_pts = np.array([[GATE_X_END, SIO2_BOTTOM], [po_x, SIO2_BOTTOM], [GATE_X_END, SIO2_BOTTOM - CH_THICK]])
            dep_rect = patches.Rectangle((GATE_X_START, SIO2_BOTTOM - CH_THICK), po_x - GATE_X_START, fc="#fbeee6", ec="#e67e22", linestyle='--', alpha=0.6)
            arr_start, arr_end = GATE_X_END - 0.2, po_x + 0.15
        ax.add_patch(MplPolygon(tri_pts, closed=True, fc=ch_color, ec=ch_edge, lw=1.2, alpha=0.9, zorder=4))
        ax.add_patch(dep_rect)
        ax.plot(po_x, SIO2_BOTTOM, 'ro', ms=7, zorder=10)
        ax.text(po_x, SIO2_BOTTOM - 0.8, "Pinch-off", ha="center", fontsize=7, color="red", fontweight="bold")
        ax.annotate("", xy=(arr_end, SIO2_BOTTOM - CH_THICK * 0.4), xytext=(arr_start, SIO2_BOTTOM - CH_THICK * 0.4), arrowprops=dict(arrowstyle='->', color=carrier_color, lw=1.4), zorder=6)
    elif "선형" in op_region:
        drain_thin = CH_THICK * (1.0 - 0.4 * (abs_vds / (vds_sat if vds_sat > 0 else 1)))
        trap_pts = np.array([[GATE_X_START, SIO2_BOTTOM], [GATE_X_END, SIO2_BOTTOM], [GATE_X_END, SIO2_BOTTOM - drain_thin], [GATE_X_START, SIO2_BOTTOM - CH_THICK]])
        ax.add_patch(MplPolygon(trap_pts, closed=True, fc=ch_color, ec=ch_edge, lw=1.2, alpha=0.85, zorder=4))
    else:
        ax.text(5, S
