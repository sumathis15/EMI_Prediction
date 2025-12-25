import pandas as pd
import numpy as np
import os


MODEL_DIR = "models"
FEATURE_COL_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")


def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Takes raw user input dictionary and returns
    model-ready dataframe aligned with training features.
    """

    # -------------------------------
    # Create DataFrame
    # -------------------------------
    df = pd.DataFrame([data])

    # -------------------------------
    # STEP 1: BASIC FEATURE ENGINEERING
    # -------------------------------
    df["total_expenses"] = (
        df["school_fees"]
        + df["college_fees"]
        + df["travel_expenses"]
        + df["groceries_utilities"]
        + df["other_monthly_expenses"]
    )

    df["disposable_income"] = (
        df["monthly_salary"]
        - df["total_expenses"]
        - df["current_emi_amount"]
    ).clip(lower=0)

    df["expense_ratio"] = df["total_expenses"] / (df["monthly_salary"] + 1)
    df["dti_ratio"] = df["current_emi_amount"] / (df["monthly_salary"] + 1)

    df["affordability_index"] = (
        df["bank_balance"] + df["emergency_fund"]
    ) / (df["requested_amount"] + 1)

    # -------------------------------
    # STEP 2: RISK & STABILITY FEATURES
    # -------------------------------
    df["credit_risk_score"] = (850 - df["credit_score"]) / 550

    df["employment_stability_score"] = np.where(
        df["years_of_employment"] >= 5, 1,
        np.where(df["years_of_employment"] >= 2, 0.5, 0.2)
    )

    df["financial_risk_index"] = (
        df["expense_ratio"] * 0.4 +
        df["dti_ratio"] * 0.4 +
        df["credit_risk_score"] * 0.2
    )

    # -------------------------------
    # STEP 3: INTERACTION FEATURES
    # -------------------------------
    df["income_credit_interaction"] = (
        df["monthly_salary"] * df["credit_risk_score"]
    )

    df["emi_income_interaction"] = (
        df["current_emi_amount"] / (df["monthly_salary"] + 1)
    )

    df["savings_buffer_ratio"] = (
        (df["bank_balance"] + df["emergency_fund"]) /
        (df["monthly_salary"] + 1)
    )

    # -------------------------------
    # STEP 4: CATEGORICAL ENCODING
    # -------------------------------
    categorical_cols = [
        "gender",
        "marital_status",
        "education",
        "employment_type",
        "company_type",
        "house_type",
        "existing_loans",
        "emi_scenario",
    ]

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # -------------------------------
    # STEP 5: COLUMN ALIGNMENT (CRITICAL)
    # -------------------------------
    expected_columns = pd.read_pickle(FEATURE_COL_PATH)

    df = df.reindex(columns=expected_columns, fill_value=0)

    return df
