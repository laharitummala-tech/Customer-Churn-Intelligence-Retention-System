import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class CustomerPreprocessor:
    def __init__(self):
        self.median_cols = [
            "Age", "Membership_Years", "Login_Frequency",
            "Session_Duration_Avg", "Cart_Abandonment_Rate",
            "Total_Purchases", "Average_Order_Value",
            "Days_Since_Last_Purchase", "Discount_Usage_Rate",
            "Returns_Rate", "Email_Open_Rate",
            "Product_Reviews_Written", "Lifetime_Value",
            "Pages_Per_Session", "Wishlist_Items",
            "Mobile_App_Usage", "Credit_Balance"
        ]

        self.mode_cols = [
            "Gender", "Country", "City",
            "Customer_Service_Calls",
            "Payment_Method_Diversity",
            "Signup_Quarter"
        ]

        self.knn_cols = ["Social_Media_Engagement_Score"]

        self.onehot_cols = ["Gender", "Country", "Signup_Quarter"]
        self.freq_cols = ["City"]

        self.median_imputer = SimpleImputer(strategy="median")
        self.mode_values = {}

        self.knn_scaler = StandardScaler()
        self.knn_imputer = KNNImputer(n_neighbors=5)

        self.onehot_encoder = OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
            sparse_output=False
        )

        self.freq_maps = {}
        self.final_columns = None

    def fix_dtypes(self, df):
        df = df.copy()

        numeric_cols = self.median_cols + self.knn_cols

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

        for col in self.mode_cols:
            df[col] = df[col].astype("object")

        return df

    def handle_invalid_values(self, df):
        df = df.copy()

        df.loc[(df["Age"] < 18) | (df["Age"] > 90), "Age"] = np.nan
        df.loc[df["Membership_Years"] < 0, "Membership_Years"] = np.nan
        df.loc[df["Login_Frequency"] < 0, "Login_Frequency"] = np.nan
        df.loc[df["Session_Duration_Avg"] < 0, "Session_Duration_Avg"] = np.nan
        df.loc[df["Pages_Per_Session"] < 0, "Pages_Per_Session"] = np.nan
        df.loc[df["Wishlist_Items"] < 0, "Wishlist_Items"] = np.nan
        df.loc[df["Total_Purchases"] < 0, "Total_Purchases"] = np.nan
        df.loc[df["Average_Order_Value"] < 0, "Average_Order_Value"] = np.nan
        df.loc[df["Days_Since_Last_Purchase"] < 0, "Days_Since_Last_Purchase"] = np.nan
        df.loc[df["Customer_Service_Calls"] < 0, "Customer_Service_Calls"] = np.nan
        df.loc[df["Product_Reviews_Written"] < 0, "Product_Reviews_Written"] = np.nan
        df.loc[df["Social_Media_Engagement_Score"] < 0, "Social_Media_Engagement_Score"] = np.nan
        df.loc[df["Mobile_App_Usage"] < 0, "Mobile_App_Usage"] = np.nan
        df.loc[df["Payment_Method_Diversity"] < 0, "Payment_Method_Diversity"] = np.nan
        df.loc[df["Lifetime_Value"] < 0, "Lifetime_Value"] = np.nan
        df.loc[df["Credit_Balance"] < 0, "Credit_Balance"] = np.nan

        percentage_cols = [
            "Cart_Abandonment_Rate",
            "Discount_Usage_Rate",
            "Returns_Rate",
            "Email_Open_Rate"
        ]

        for col in percentage_cols:
            df.loc[(df[col] < 0) | (df[col] > 100), col] = np.nan

        df.loc[~df["Gender"].isin(["Male", "Female", "Other"]), "Gender"] = np.nan
        df.loc[~df["Signup_Quarter"].isin(["Q1", "Q2", "Q3", "Q4"]), "Signup_Quarter"] = np.nan

        return df

    def encode_features(self, X):
        X = X.copy()

        onehot_array = self.onehot_encoder.transform(X[self.onehot_cols])
        onehot_columns = self.onehot_encoder.get_feature_names_out(self.onehot_cols)

        onehot_df = pd.DataFrame(
            onehot_array,
            columns=onehot_columns,
            index=X.index
        )

        for col in self.freq_cols:
            X[f"{col}_freq"] = X[col].map(self.freq_maps[col]).fillna(0)

        X = X.drop(columns=self.onehot_cols + self.freq_cols)
        X = pd.concat([X, onehot_df], axis=1)

        return X

    def fit(self, X):
        X = X.copy()

        if "Churned" in X.columns:
            X = X.drop("Churned", axis=1)

        X = self.fix_dtypes(X)
        X = self.handle_invalid_values(X)

        self.median_imputer.fit(X[self.median_cols])

        for col in self.mode_cols:
            self.mode_values[col] = X[col].mode(dropna=True)[0]

        knn_scaled = self.knn_scaler.fit_transform(X[self.knn_cols])
        self.knn_imputer.fit(knn_scaled)

        X[self.median_cols] = self.median_imputer.transform(X[self.median_cols])

        for col in self.mode_cols:
            X[col] = X[col].fillna(self.mode_values[col])

        knn_imputed = self.knn_imputer.transform(knn_scaled)
        knn_original = self.knn_scaler.inverse_transform(knn_imputed)

        X[self.knn_cols] = pd.DataFrame(
            knn_original,
            columns=self.knn_cols,
            index=X.index
        )

        self.onehot_encoder.fit(X[self.onehot_cols])

        for col in self.freq_cols:
            self.freq_maps[col] = X[col].value_counts()

        X_encoded = self.encode_features(X)
        self.final_columns = X_encoded.columns.tolist()

        return self

    def transform(self, X):
        X = X.copy()

        if "Churned" in X.columns:
            X = X.drop("Churned", axis=1)

        X = self.fix_dtypes(X)
        X = self.handle_invalid_values(X)

        X[self.median_cols] = self.median_imputer.transform(X[self.median_cols])

        for col in self.mode_cols:
            X[col] = X[col].fillna(self.mode_values[col])

        knn_scaled = self.knn_scaler.transform(X[self.knn_cols])
        knn_imputed = self.knn_imputer.transform(knn_scaled)
        knn_original = self.knn_scaler.inverse_transform(knn_imputed)

        X[self.knn_cols] = pd.DataFrame(
            knn_original,
            columns=self.knn_cols,
            index=X.index
        )

        X_encoded = self.encode_features(X)

        X_encoded = X_encoded.reindex(
            columns=self.final_columns,
            fill_value=0
        )

        return X_encoded

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)