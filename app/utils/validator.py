import pandas as pd


class DataValidator:
    def __init__(self):
        self.expected_columns = [
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

        self.rules = {
            "Age < 18 or Age > 90": lambda df: ((df["Age"] < 18) | (df["Age"] > 90)).sum(),
            "Total_Purchases < 0": lambda df: (df["Total_Purchases"] < 0).sum(),
            "Cart_Abandonment_Rate > 100": lambda df: (df["Cart_Abandonment_Rate"] > 100).sum(),
            "Discount_Usage_Rate > 100": lambda df: (df["Discount_Usage_Rate"] > 100).sum(),
            "Returns_Rate > 100": lambda df: (df["Returns_Rate"] > 100).sum(),
            "Email_Open_Rate > 100": lambda df: (df["Email_Open_Rate"] > 100).sum(),
            "Customer_Service_Calls < 0": lambda df: (df["Customer_Service_Calls"] < 0).sum(),
            "Wishlist_Items < 0": lambda df: (df["Wishlist_Items"] < 0).sum(),
            "Product_Reviews_Written < 0": lambda df: (df["Product_Reviews_Written"] < 0).sum(),
            "Mobile_App_Usage < 0": lambda df: (df["Mobile_App_Usage"] < 0).sum(),
            "Lifetime_Value < 0": lambda df: (df["Lifetime_Value"] < 0).sum(),
            "Credit_Balance < 0": lambda df: (df["Credit_Balance"] < 0).sum()
        }

    def remove_target_column(self, df):
        df = df.copy()

        if "Churned" in df.columns:
            df = df.drop("Churned", axis=1)

        return df

    def column_validation(self, df):
        missing_cols = [col for col in self.expected_columns if col not in df.columns]
        extra_cols = [col for col in df.columns if col not in self.expected_columns]

        return missing_cols, extra_cols

    def missing_report(self, df):
        missing_df = pd.DataFrame({
            "Column": df.columns,
            "Missing Count": df.isnull().sum().values,
            "Missing Percentage": (df.isnull().mean() * 100).round(2).values
        })

        return missing_df[missing_df["Missing Count"] > 0]

    def invalid_report(self, df):
        report = []

        for rule_name, rule in self.rules.items():
            try:
                invalid_count = rule(df)

                if invalid_count > 0:
                    report.append({
                        "Invalid Rule": rule_name,
                        "Invalid Count": invalid_count
                    })

            except KeyError:
                continue

        return pd.DataFrame(report)

    def validate(self, df):
        df = self.remove_target_column(df)

        missing_cols, extra_cols = self.column_validation(df)

        result = {
            "clean_df": df,
            "missing_columns": missing_cols,
            "extra_columns": extra_cols,
            "missing_report": pd.DataFrame(),
            "invalid_report": pd.DataFrame(),
            "is_valid": True,
            "message": "Validation completed successfully"
        }

        if len(missing_cols) > 0:
            result["is_valid"] = False
            result["message"] = "Missing required columns found"
            return result

        result["missing_report"] = self.missing_report(df)
        result["invalid_report"] = self.invalid_report(df)

        return result