import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Customer Churn Intelligence System",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Intelligence System")
st.write("Upload customer data, validate quality, clean issues, and predict churn risk.")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # 1. Overview
    st.header("1. Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Duplicate Rows", df.duplicated().sum())
    col4.metric("Missing Cells", df.isnull().sum().sum())

    # 2. Preview
    st.header("2. Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    # 3. Column Validation
    st.header("3. Column Validation")

    expected_columns = [
        "Age", "Gender", "Country", "City", "Membership_Years",
        "Login_Frequency", "Session_Duration_Avg", "Pages_Per_Session",
        "Cart_Abandonment_Rate", "Wishlist_Items", "Total_Purchases",
        "Average_Order_Value", "Days_Since_Last_Purchase",
        "Discount_Usage_Rate", "Returns_Rate", "Email_Open_Rate",
        "Customer_Service_Calls", "Product_Reviews_Written",
        "Social_Media_Engagement_Score", "Mobile_App_Usage",
        "Payment_Method_Diversity", "Lifetime_Value", "Credit_Balance",
        "Signup_Quarter"
    ]

    missing_cols = [col for col in expected_columns if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in expected_columns]

    if len(missing_cols) == 0:
        st.success("All required columns are present.")
    else:
        st.error("Missing required columns:")
        st.write(missing_cols)

    if len(extra_cols) > 0:
        st.warning("Extra columns found:")
        st.write(extra_cols)

    # 4. Duplicate Rows
    st.header("4. Duplicate Rows")

    duplicate_count = df.duplicated().sum()
    st.write("Duplicate rows found:", duplicate_count)

    if duplicate_count > 0:
        st.dataframe(df[df.duplicated()].head(20), use_container_width=True)

    # 5. Missing Values
    st.header("5. Missing Values Report")

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": df.isnull().sum().values,
        "Missing Percentage": (df.isnull().mean() * 100).round(2).values
    })

    missing_df = missing_df[missing_df["Missing Count"] > 0]

    if missing_df.empty:
        st.success("No missing values found.")
    else:
        st.dataframe(missing_df, use_container_width=True)

    # 6. Invalid Values
    st.header("6. Invalid Values Report")

    invalid_report = {}

    if "Age" in df.columns:
        invalid_report["Age < 18 or Age > 90"] = ((df["Age"] < 18) | (df["Age"] > 90)).sum()

    if "Total_Purchases" in df.columns:
        invalid_report["Total_Purchases < 0"] = (df["Total_Purchases"] < 0).sum()

    if "Cart_Abandonment_Rate" in df.columns:
        invalid_report["Cart_Abandonment_Rate > 100"] = (df["Cart_Abandonment_Rate"] > 100).sum()

    if "Discount_Usage_Rate" in df.columns:
        invalid_report["Discount_Usage_Rate > 100"] = (df["Discount_Usage_Rate"] > 100).sum()

    if "Returns_Rate" in df.columns:
        invalid_report["Returns_Rate > 100"] = (df["Returns_Rate"] > 100).sum()

    invalid_df = pd.DataFrame(
        invalid_report.items(),
        columns=["Invalid Rule", "Invalid Count"]
    )

    st.dataframe(invalid_df, use_container_width=True)

    st.header("7. Prediction")
    st.info("Prediction section will be added after connecting saved model artifacts.")

else:
    st.info("Upload a CSV file from the sidebar to begin.")