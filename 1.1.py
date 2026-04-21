import streamlit as st
st.set_page_config(page_title="叶菜需求预测平台",page_icon="🥬",layout="wide",initial_sidebar_state="expanded")
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------- 【新增】把天气选择框鼠标改成小抓手 ----------------------
st.markdown("""
    <style>
        /* 目标：天气选择框的鼠标样式 */
        [data-testid="stSelectbox"] > div > div {
            cursor: pointer !important;
        }
        /* 同时优化下拉选项的鼠标样式 */
        [data-baseweb="select"] {
            cursor: pointer !important;
        }
        /* 下拉箭头也改成小抓手 */
        [data-testid="stSelectbox"] svg {
            cursor: pointer !important;
        }
    </style>
""", unsafe_allow_html=True)

# 全局主题颜色
MAIN_COLOR = "#FF4B4B"
INIT_LINE_COLOR = "#4ECDC4"
CURRENT_LINE_COLOR = "#FF0000"

# ---------------------- 核心参数 ----------------------
REGRESSION_WEIGHTS = {
    "temp": -0.02,  # 温度每升1℃，销量降0.02吨
    "rain": -0.01,  # 降水量每升1mm，销量降0.01吨
    "holiday": 0.15,  # 周末/节假日，销量升0.15吨
    "discount": 0.8  # 折扣每降0.1，销量升0.08吨
}

WEATHER_RAIN_MAP = {
    "晴": 0.2,
    "多云": 0.7,
    "阴": 3.0,
    "雨": 7.5
}

BASE_DEMAND = 5.0
LOSS_RATE = 0.1116


# ---------------------- 核心计算函数 ----------------------
@st.cache_data
def calculate_demand(temp, weather, is_holiday, discount):
    rain = WEATHER_RAIN_MAP[weather]

    if 20 <= temp <= 25:
        temp_mem = 1.0
    elif temp < 20:
        temp_mem = max(0, 1 - (20 - temp) / 30)
    else:
        temp_mem = max(0, 1 - (temp - 25) / 15)

    weather_mem = {
        "晴": 0.0,
        "多云": 0.1,
        "阴": 0.3,
        "雨": 0.8
    }[weather]

    weather_temp_coef = (temp_mem + weather_mem) / 2

    base_pred = BASE_DEMAND
    base_pred += REGRESSION_WEIGHTS["temp"] * (temp - 20)
    base_pred += REGRESSION_WEIGHTS["rain"] * rain
    base_pred += REGRESSION_WEIGHTS["holiday"] * (1 if is_holiday else 0)
    base_pred += REGRESSION_WEIGHTS["discount"] * (1 - discount)

    final_pred = base_pred * (0.8 + 0.4 * weather_temp_coef)
    return max(0, final_pred)


@st.cache_data
def calculate_replenishment(pred_demand, current_stock):
    safety_stock = pred_demand * LOSS_RATE
    replenishment = pred_demand - current_stock + safety_stock
    return max(0, replenishment)


# ---------------------- 侧边栏参数输入 ----------------------
with st.sidebar:
    st.markdown("## 🥬 叶菜需求预测平台")
    st.markdown("---")
    st.markdown("### 📝 参数输入")

    # 天气选择（鼠标已改成小抓手）
    weather = st.selectbox(
        "天气状况",
        ["晴", "多云", "阴", "雨"],
        index=0
    )

    # 平均温度
    temp = st.slider(
        "平均温度 (℃)",
        min_value=-10,
        max_value=40,
        value=20,
        step=1
    )

    # 折扣系数
    discount = st.slider(
        "折扣系数 (0.5=5折, 1.0=原价)",
        min_value=0.5,
        max_value=1.0,
        value=0.9,
        step=0.05
    )

    # 是否周末/节假日
    is_holiday = st.radio(
        "是否为周末/节假日",
        ["否", "是"],
        index=1
    )
    is_holiday_bool = True if is_holiday == "是" else False

    # 当前库存
    current_stock = st.number_input(
        "当前期初库存 (吨)",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1
    )

    st.markdown("---")
    predict_button = st.button("📊 生成预测结果", type="primary", use_container_width=True)

# ---------------------- 主页面内容 ----------------------
st.markdown("# 🥬 叶菜需求量动态预测与补货可视化平台")
st.markdown("---")

if predict_button or 'pred_demand' in st.session_state:
    if predict_button:
        st.session_state.initial_weather = weather
        st.session_state.initial_temp = temp
        st.session_state.initial_discount = discount
        st.session_state.initial_is_holiday = is_holiday_bool
        st.session_state.initial_stock = current_stock

        st.session_state.pred_demand = calculate_demand(temp, weather, is_holiday_bool, discount)
        st.session_state.replenishment = calculate_replenishment(st.session_state.pred_demand, current_stock)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="📊 预测需求量",
            value=f"{st.session_state.pred_demand:.2f} 吨"
        )
    with col2:
        st.metric(
            label="📦 推荐补货量",
            value=f"{st.session_state.replenishment:.2f} 吨"
        )
    with col3:
        st.metric(
            label="⚠️ 安全库存",
            value=f"{st.session_state.pred_demand * LOSS_RATE:.2f} 吨"
        )

    st.markdown("---")

    tab1, tab2 = st.tabs(["📈 单因素敏感性分析", "🔄 多因素情景模拟"])

    # ---------------------- Tab1：单因素敏感性分析 ----------------------
    with tab1:
        st.markdown("## 单因素敏感性分析")
        st.caption("规则：固定初始基准参数，选择要分析的因素，拖动滑块查看单因素对需求的独立影响")

        factor = st.radio(
            "选择要分析的影响因素",
            ["🌡️ 平均温度", "💧 降水量", "💰 折扣系数", "📦 现有库存"],
            index=0,
            horizontal=True
        )

        if factor == "🌡️ 平均温度":
            st.markdown("---")
            st.markdown("### 🌡️ 平均温度对叶菜需求量的影响")

            temp_range = np.arange(-10, 41, 1)
            temp_demand_list = []
            for t in temp_range:
                d = calculate_demand(
                    t,
                    st.session_state.initial_weather,
                    st.session_state.initial_is_holiday,
                    st.session_state.initial_discount
                )
                temp_demand_list.append(d)

            current_temp = st.slider(
                "拖动调整温度 (℃)",
                min_value=-10,
                max_value=40,
                value=st.session_state.initial_temp,
                step=1,
                key="temp_slider"
            )

            current_temp_demand = calculate_demand(
                current_temp,
                st.session_state.initial_weather,
                st.session_state.initial_is_holiday,
                st.session_state.initial_discount
            )
            current_temp_replenishment = calculate_replenishment(current_temp_demand, st.session_state.initial_stock)

            temp_df = pd.DataFrame({
                "平均温度 (℃)": temp_range,
                "预测需求量 (吨)": temp_demand_list
            })
            fig_temp = px.line(
                temp_df,
                x="平均温度 (℃)",
                y="预测需求量 (吨)",
                title="平均温度对叶菜需求量的影响",
                markers=True,
                color_discrete_sequence=[MAIN_COLOR]
            )
            fig_temp.add_scatter(
                x=[current_temp],
                y=[current_temp_demand],
                mode="markers",
                marker=dict(color=CURRENT_LINE_COLOR, size=14, symbol="circle"),
                name="当前值"
            )
            fig_temp.add_vline(
                x=st.session_state.initial_temp,
                line_dash="dash",
                line_color=INIT_LINE_COLOR,
                line_width=2,
                annotation_text="初始温度"
            )
            fig_temp.add_vline(
                x=current_temp,
                line_dash="dot",
                line_color=CURRENT_LINE_COLOR,
                line_width=2,
                annotation_text="当前温度"
            )
            fig_temp.update_layout(
                hovermode="x unified",
                clickmode="event+select",
                plot_bgcolor="#f8f9fa",
                paper_bgcolor="#ffffff",
                font=dict(size=14)
            )
            st.plotly_chart(fig_temp, use_container_width=True, config={'displayModeBar': False})

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(
                    label="当前温度对应的需求量",
                    value=f"{current_temp_demand:.2f} 吨",
                    delta=f"{current_temp_demand - st.session_state.pred_demand:+.2f} 吨"
                )
            with col_b:
                st.metric(
                    label="当前温度对应的推荐补货量",
                    value=f"{current_temp_replenishment:.2f} 吨",
                    delta=f"{current_temp_replenishment - st.session_state.replenishment:+.2f} 吨"
                )
            st.info(
                f"💡 线性回归权重提示：温度每上升 1℃，叶菜需求量预计 **下降 {abs(REGRESSION_WEIGHTS['temp']):.2f} 吨**")

        elif factor == "💧 降水量":
            st.markdown("---")
            st.markdown("### 💧 降水量对叶菜需求量的影响")

            rain_range = np.arange(0, 21, 1)
            rain_demand_list = []
            for r in rain_range:
                if r < 1:
                    w = "晴"
                elif r < 2:
                    w = "多云"
                elif r < 5:
                    w = "阴"
                else:
                    w = "雨"
                d = calculate_demand(
                    st.session_state.initial_temp,
                    w,
                    st.session_state.initial_is_holiday,
                    st.session_state.initial_discount
                )
                rain_demand_list.append(d)

            current_rain = st.slider(
                "拖动调整降水量 (mm)",
                min_value=0,
                max_value=20,
                value=int(WEATHER_RAIN_MAP[st.session_state.initial_weather]),
                step=1,
                key="rain_slider"
            )

            if current_rain < 1:
                current_weather = "晴"
            elif current_rain < 2:
                current_weather = "多云"
            elif current_rain < 5:
                current_weather = "阴"
            else:
                current_weather = "雨"
            current_rain_demand = calculate_demand(
                st.session_state.initial_temp,
                current_weather,
                st.session_state.initial_is_holiday,
                st.session_state.initial_discount
            )
            current_rain_replenishment = calculate_replenishment(current_rain_demand, st.session_state.initial_stock)

            rain_df = pd.DataFrame({
                "降水量 (mm)": rain_range,
                "预测需求量 (吨)": rain_demand_list
            })
            fig_rain = px.line(
                rain_df,
                x="降水量 (mm)",
                y="预测需求量 (吨)",
                title="降水量对叶菜需求量的影响",
                markers=True,
                color_discrete_sequence=["#45B7D1"]
            )
            fig_rain.add_scatter(
                x=[current_rain],
                y=[current_rain_demand],
                mode="markers",
                marker=dict(color=CURRENT_LINE_COLOR, size=14, symbol="circle"),
                name="当前值"
            )
            fig_rain.add_vline(
                x=WEATHER_RAIN_MAP[st.session_state.initial_weather],
                line_dash="dash",
                line_color=INIT_LINE_COLOR,
                line_width=2,
                annotation_text="初始降水量"
            )
            fig_rain.add_vline(
                x=current_rain,
                line_dash="dot",
                line_color=CURRENT_LINE_COLOR,
                line_width=2,
                annotation_text="当前降水量"
            )
            fig_rain.update_layout(
                hovermode="x unified",
                clickmode="event+select",
                plot_bgcolor="#f8f9fa",
                paper_bgcolor="#ffffff",
                font=dict(size=14)
            )
            st.plotly_chart(fig_rain, use_container_width=True, config={'displayModeBar': False})

            col_c, col_d = st.columns(2)
            with col_c:
                st.metric(
                    label="当前降水量对应的需求量",
                    value=f"{current_rain_demand:.2f} 吨",
                    delta=f"{current_rain_demand - st.session_state.pred_demand:+.2f} 吨"
                )
            with col_d:
                st.metric(
                    label="当前降水量对应的推荐补货量",
                    value=f"{current_rain_replenishment:.2f} 吨",
                    delta=f"{current_rain_replenishment - st.session_state.replenishment:+.2f} 吨"
                )
            st.info(
                f"💡 线性回归权重提示：降水量每上升 1mm，叶菜需求量预计 **下降 {abs(REGRESSION_WEIGHTS['rain']):.2f} 吨**")

        elif factor == "💰 折扣系数":
            st.markdown("---")
            st.markdown("### 💰 折扣系数对叶菜需求量的影响")

            discount_range = np.arange(0.5, 1.05, 0.05)
            discount_demand_list = []
            for d in discount_range:
                dem = calculate_demand(
                    st.session_state.initial_temp,
                    st.session_state.initial_weather,
                    st.session_state.initial_is_holiday,
                    d
                )
                discount_demand_list.append(dem)

            current_discount = st.slider(
                "拖动调整折扣系数",
                min_value=0.5,
                max_value=1.0,
                value=st.session_state.initial_discount,
                step=0.05,
                key="discount_slider"
            )

            current_discount_demand = calculate_demand(
                st.session_state.initial_temp,
                st.session_state.initial_weather,
                st.session_state.initial_is_holiday,
                current_discount
            )
            current_discount_replenishment = calculate_replenishment(current_discount_demand,
                                                                     st.session_state.initial_stock)

            discount_df = pd.DataFrame({
                "折扣系数": discount_range,
                "预测需求量 (吨)": discount_demand_list
            })
            fig_discount = px.line(
                discount_df,
                x="折扣系数",
                y="预测需求量 (吨)",
                title="折扣系数对叶菜需求量的影响",
                markers=True,
                color_discrete_sequence=["#4ECDC4"]
            )
            fig_discount.add_scatter(
                x=[current_discount],
                y=[current_discount_demand],
                mode="markers",
                marker=dict(color=CURRENT_LINE_COLOR, size=14, symbol="circle"),
                name="当前值"
            )
            fig_discount.add_vline(
                x=st.session_state.initial_discount,
                line_dash="dash",
                line_color=INIT_LINE_COLOR,
                line_width=2,
                annotation_text="初始折扣"
            )
            fig_discount.add_vline(
                x=current_discount,
                line_dash="dot",
                line_color=CURRENT_LINE_COLOR,
                line_width=2,
                annotation_text="当前折扣"
            )
            fig_discount.update_xaxes(autorange="reversed")
            fig_discount.update_layout(
                hovermode="x unified",
                clickmode="event+select",
                plot_bgcolor="#f8f9fa",
                paper_bgcolor="#ffffff",
                font=dict(size=14)
            )
            st.plotly_chart(fig_discount, use_container_width=True, config={'displayModeBar': False})

            col_e, col_f = st.columns(2)
            with col_e:
                st.metric(
                    label="当前折扣对应的需求量",
                    value=f"{current_discount_demand:.2f} 吨",
                    delta=f"{current_discount_demand - st.session_state.pred_demand:+.2f} 吨"
                )
            with col_f:
                st.metric(
                    label="当前折扣对应的推荐补货量",
                    value=f"{current_discount_replenishment:.2f} 吨",
                    delta=f"{current_discount_replenishment - st.session_state.replenishment:+.2f} 吨"
                )
            st.info(
                f"💡 线性回归权重提示：折扣每降低 0.1（多打1折），叶菜需求量预计 **上升 {REGRESSION_WEIGHTS['discount'] * 0.1:.2f} 吨**")

        else:
            st.markdown("---")
            st.markdown("### 📦 现有库存对推荐补货量的影响")

            stock_range = np.arange(0, 10.1, 0.5)
            stock_replenishment_list = []
            for s in stock_range:
                r = calculate_replenishment(st.session_state.pred_demand, s)
                stock_replenishment_list.append(r)

            current_stock_slider = st.slider(
                "拖动调整现有库存 (吨)",
                min_value=0.0,
                max_value=10.0,
                value=st.session_state.initial_stock,
                step=0.5,
                key="stock_slider"
            )

            current_stock_replenishment = calculate_replenishment(st.session_state.pred_demand, current_stock_slider)

            stock_df = pd.DataFrame({
                "现有库存 (吨)": stock_range,
                "推荐补货量 (吨)": stock_replenishment_list
            })
            fig_stock = px.line(
                stock_df,
                x="现有库存 (吨)",
                y="推荐补货量 (吨)",
                title="现有库存对推荐补货量的影响",
                markers=True,
                color_discrete_sequence=["#96CEB4"]
            )
            fig_stock.add_scatter(
                x=[current_stock_slider],
                y=[current_stock_replenishment],
                mode="markers",
                marker=dict(color=CURRENT_LINE_COLOR, size=14, symbol="circle"),
                name="当前值"
            )
            fig_stock.add_vline(
                x=st.session_state.initial_stock,
                line_dash="dash",
                line_color=INIT_LINE_COLOR,
                line_width=2,
                annotation_text="初始库存"
            )
            fig_stock.add_vline(
                x=current_stock_slider,
                line_dash="dot",
                line_color=CURRENT_LINE_COLOR,
                line_width=2,
                annotation_text="当前库存"
            )
            fig_stock.update_layout(
                hovermode="x unified",
                clickmode="event+select",
                plot_bgcolor="#f8f9fa",
                paper_bgcolor="#ffffff",
                font=dict(size=14)
            )
            st.plotly_chart(fig_stock, use_container_width=True, config={'displayModeBar': False})

            st.metric(
                label="当前库存对应的推荐补货量",
                value=f"{current_stock_replenishment:.2f} 吨",
                delta=f"{current_stock_replenishment - st.session_state.replenishment:+.2f} 吨"
            )
            st.info("💡 说明：现有库存仅影响推荐补货量，不影响叶菜需求量预测")

    # ---------------------- Tab2：多因素情景模拟 ----------------------
    with tab2:
        st.markdown("## 多因素情景模拟")
        st.caption("规则：可自由调整所有经营参数，模拟多因素叠加后的综合需求与补货结果")

        col_a, col_b = st.columns(2)
        with col_a:
            new_weather = st.selectbox(
                "调整天气状况",
                ["晴", "多云", "阴", "雨"],
                index=["晴", "多云", "阴", "雨"].index(st.session_state.initial_weather)
            )
            new_temp = st.slider(
                "调整平均温度 (℃)",
                min_value=-10,
                max_value=40,
                value=st.session_state.initial_temp,
                step=1
            )
            new_discount = st.slider(
                "调整折扣系数",
                min_value=0.5,
                max_value=1.0,
                value=st.session_state.initial_discount,
                step=0.05
            )
        with col_b:
            new_is_holiday = st.radio(
                "调整是否为周末/节假日",
                ["否", "是"],
                index=1 if st.session_state.initial_is_holiday else 0
            )
            new_is_holiday_bool = True if new_is_holiday == "是" else False
            new_stock = st.number_input(
                "调整现有库存 (吨)",
                min_value=0.0,
                max_value=10.0,
                value=st.session_state.initial_stock,
                step=0.1
            )

        new_pred = calculate_demand(new_temp, new_weather, new_is_holiday_bool, new_discount)
        new_replenishment = calculate_replenishment(new_pred, new_stock)

        col_c, col_d = st.columns(2)
        with col_c:
            st.metric(
                label="📊 新情景预测需求量",
                value=f"{new_pred:.2f} 吨",
                delta=f"{new_pred - st.session_state.pred_demand:+.2f} 吨"
            )
        with col_d:
            st.metric(
                label="📦 新情景推荐补货量",
                value=f"{new_replenishment:.2f} 吨",
                delta=f"{new_replenishment - st.session_state.replenishment:+.2f} 吨"
            )

        st.markdown("---")
        st.markdown("### 情景对比可视化")

        comparison_df = pd.DataFrame({
            "情景": ["初始基准情景", "新调整情景"],
            "预测需求量 (吨)": [st.session_state.pred_demand, new_pred],
            "推荐补货量 (吨)": [st.session_state.replenishment, new_replenishment]
        })

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=comparison_df["情景"],
            y=comparison_df["预测需求量 (吨)"],
            name="预测需求量",
            marker_color=MAIN_COLOR,
            text=comparison_df["预测需求量 (吨)"].round(2),
            textposition="auto",
            width=0.25
        ))
        fig.add_trace(go.Bar(
            x=comparison_df["情景"],
            y=comparison_df["推荐补货量 (吨)"],
            name="推荐补货量",
            marker_color="#4ECDC4",
            text=comparison_df["推荐补货量 (吨)"].round(2),
            textposition="auto",
            width=0.25
        ))
        fig.update_layout(
            title="初始情景 vs 新情景对比",
            barmode="group",
            bargap=0.4,
            xaxis_title="情景",
            yaxis_title="吨",
            legend_title="指标",
            plot_bgcolor="#f8f9fa",
            paper_bgcolor="#ffffff",
            font=dict(size=14)
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 请在左侧侧边栏输入参数，点击【生成预测结果】按钮开始使用")

    st.markdown("---")
    st.markdown("### 📖 平台功能介绍")
    col_e, col_f = st.columns(2)
    with col_e:
        st.markdown("""
        **📈 单因素敏感性分析**
        - 固定其他参数，选择要分析的因素
        - 拖动滑块实时查看单因素影响，响应丝滑
        - 点击折线图任意位置显示对应数值
        - 配套展示多元线性回归权重提示
        """)
    with col_f:
        st.markdown("""
        **🔄 多因素情景模拟**
        - 可自由调整所有经营参数
        - 模拟多因素叠加后的综合需求变化
        - 对比初始情景与新情景的差异
        - 贴合真实经营场景，辅助商户决策
        """)

