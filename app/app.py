import streamlit as st
import pandas as pd

from utils.validator import DataValidator
from utils.preprocessing import preprocess_data
from utils.prediction import predict_churn, risk_segment
from utils.shap_explainer import get_top_churn_reasons_batched
@st.cache_data(show_spinner=False)
def cached_shap_reasons(processed_df):
    return get_top_churn_reasons_batched(
        processed_df,
        top_n=3,
        batch_size=200
    )

st.set_page_config(
    page_title="Customer Churn Intelligence System",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Intelligence System")
st.write("Upload customer data, validate quality, clean issues, and predict churn risk.")

validator = DataValidator()

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    validation_result = validator.validate(df)
    clean_df = validation_result["clean_df"].copy()
    duplicate_count = clean_df.duplicated().sum()

    clean_df.insert(0, "Customer_ID", range(1, len(clean_df) + 1))

    st.header("1. Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", clean_df.shape[0])
    col2.metric("Columns", clean_df.shape[1])
    col3.metric("Duplicate Rows", duplicate_count)
    col4.metric("Missing Cells", clean_df.isnull().sum().sum())

    st.header("2. Dataset Preview")
    st.dataframe(clean_df.head(20), use_container_width=True)

    st.header("3. Column Validation")

    missing_cols = validation_result["missing_columns"]
    extra_cols = validation_result["extra_columns"]

    if len(missing_cols) == 0:
        st.success("All required columns are present.")
    else:
        st.error("Missing required columns:")
        st.write(missing_cols)

    if len(extra_cols) > 0:
        st.warning("Extra columns found:")
        st.write(extra_cols)

    st.header("4. Missing Values Report")

    missing_df = validation_result["missing_report"]

    if missing_df.empty:
        st.success("No missing values found.")
    else:
        st.dataframe(missing_df, use_container_width=True)

    st.header("5. Invalid Values Report")

    invalid_df = validation_result["invalid_report"]

    if invalid_df.empty:
        st.success("No invalid values found.")
    else:
        col1, col2 = st.columns(2)
        col1.metric("Invalid Rules Triggered", invalid_df.shape[0])
        col2.metric("Total Invalid Values", invalid_df["Invalid Count"].sum())
        st.dataframe(invalid_df, use_container_width=True)

    if validation_result["is_valid"]:
        st.header("6. Data Preprocessing")

        model_input_df = clean_df.drop(columns=["Customer_ID"])
        processed_df = preprocess_data(model_input_df)

        st.success("Invalid values handled.")
        st.success("Missing values imputed.")
        st.success("Categorical encoding completed.")
        st.success("Data is ready for prediction.")

        st.write("Processed data shape:", processed_df.shape)

        predictions, probabilities = predict_churn(processed_df)

        final_df = clean_df.copy()
        final_df["Churn_Prediction"] = predictions
        final_df["Churn_Probability"] = probabilities
        final_df["Risk_Segment"] = final_df["Churn_Probability"].apply(risk_segment)

        total_revenue = final_df["Lifetime_Value"].sum()
        revenue_at_risk = (final_df["Lifetime_Value"] * final_df["Churn_Probability"]).sum()

        high_risk_count = (final_df["Risk_Segment"] == "High Risk").sum()
        medium_risk_count = (final_df["Risk_Segment"] == "Medium Risk").sum()
        low_risk_count = (final_df["Risk_Segment"] == "Low Risk").sum()

        st.header("7. Business Impact Dashboard")

        st.markdown("""
        <style>
        .metric-card {
            background: linear-gradient(135deg, #ffffff, #f8f9ff);
            padding: 22px;
            border-radius: 18px;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
            text-align: center;
            border: 1px solid #eeeeee;
        }
        .metric-title {
            font-size: 15px;
            color: #6c757d;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .metric-value {
            font-size: 28px;
            color: #222;
            font-weight: 800;
        }
        </style>
        """, unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">💰 Total Revenue</div>
            <div class="metric-value">${total_revenue:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

        col2.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">⚠️ Revenue At Risk</div>
            <div class="metric-value">${revenue_at_risk:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

        col3.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🔴 High Risk</div>
            <div class="metric-value">{high_risk_count}</div>
        </div>
        """, unsafe_allow_html=True)

        col4.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🟠 Medium Risk</div>
            <div class="metric-value">{medium_risk_count}</div>
        </div>
        """, unsafe_allow_html=True)

        col5.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🟢 Low Risk</div>
            <div class="metric-value">{low_risk_count}</div>
        </div>
        """, unsafe_allow_html=True)

        st.header("8. Customer Risk Overview")

        risk_filter = st.multiselect(
            "Filter by Risk Segment",
            options=["High Risk", "Medium Risk", "Low Risk"],
            default=["High Risk", "Medium Risk", "Low Risk"]
        )

        overview_df = final_df[final_df["Risk_Segment"].isin(risk_filter)]

        overview_cols = [
            "Customer_ID",
            "Churn_Prediction",
            "Churn_Probability",
            "Risk_Segment"
        ]

        st.dataframe(
            overview_df[overview_cols],
            use_container_width=True
        )

        overview_csv = overview_df[overview_cols].to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Risk Overview",
            data=overview_csv,
            file_name="customer_risk_overview.csv",
            mime="text/csv"
        )

        
        
        st.header("9. High Risk Customer Action Center")

        high_risk_df = final_df[final_df["Risk_Segment"] == "High Risk"].copy()
        high_risk_processed_df = processed_df.loc[high_risk_df.index]

        st.write(f"High Risk Customers Found: {high_risk_df.shape[0]}")

        st.warning(
            "Generating SHAP reasons for all high-risk customers may take some time."
        )

        if "high_risk_shap_df" not in st.session_state:
            st.session_state.high_risk_shap_df = None

        if st.button("Generate SHAP Reasons for All High Risk Customers"):
            with st.spinner("Generating SHAP reasons in batches..."):
                temp_df = high_risk_df.copy()
                temp_df["Top_3_Churn_Reasons"] = cached_shap_reasons(
                    high_risk_processed_df
                )

                st.session_state.high_risk_shap_df = temp_df

            st.success("SHAP reasons generated successfully.")

        if st.session_state.high_risk_shap_df is not None:
            shap_df = st.session_state.high_risk_shap_df.copy()

            reason_options = sorted(
                set(
                    reason.strip()
                    for reasons in shap_df["Top_3_Churn_Reasons"]
                    for reason in reasons.split(",")
                )
            )

            selected_reasons = st.multiselect(
                "Filter High Risk Customers by SHAP Reason",
                options=reason_options
            )

            if selected_reasons:
                shap_df = shap_df[
                    shap_df["Top_3_Churn_Reasons"].apply(
                        lambda x: any(reason in x for reason in selected_reasons)
                    )
                ]

            action_cols = [
                "Customer_ID",
                "Churn_Prediction",
                "Churn_Probability",
                "Risk_Segment",
                "Top_3_Churn_Reasons"
            ]

            st.dataframe(
                shap_df[action_cols],
                use_container_width=True
            )

            high_risk_csv = shap_df[action_cols].to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download High Risk Customers with SHAP Reasons",
                data=high_risk_csv,
                file_name="high_risk_customers_with_reasons.csv",
                mime="text/csv"
            )
        

        

    else:
        st.error("Fix missing columns before prediction.")

else:
    st.info("Upload a CSV file from the sidebar to begin.")