import streamlit as st
import pandas as pd

from utils.validator import DataValidator
from utils.preprocessing import preprocess_data
from utils.prediction import predict_churn, risk_segment


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

        st.header("7. Prediction Results")

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
        
        st.subheader("Filter Customers by Risk Segment")
        
        
        risk_filter = st.multiselect(
            "Select Risk Segment",
            options=["High Risk", "Medium Risk", "Low Risk"],
            default=["High Risk", "Medium Risk", "Low Risk"]
        )

        filtered_df = final_df[final_df["Risk_Segment"].isin(risk_filter)]

        st.dataframe(
            filtered_df[[
                "Customer_ID",
                "Churn_Prediction",
                "Churn_Probability",
                "Risk_Segment"
            ]],
            use_container_width=True
        )

        
    else:
        st.error("Fix missing columns before prediction.")

else:
    st.info("Upload a CSV file from the sidebar to begin.")