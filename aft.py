
from lifelines import (
    LogLogisticAFTFitter,
    LogNormalAFTFitter,
    WeibullAFTFitter
)
import matplotlib.pyplot as plt
ll_aft = LogLogisticAFTFitter()
ln_aft = LogNormalAFTFitter()
wb_aft = WeibullAFTFitter()

from config import df

# # 3️⃣ Fit models
print("⏳ Fitting AFT models...")
ll_aft.fit(df, duration_col='week', event_col='arrest')
ln_aft.fit(df, duration_col='week', event_col='arrest')
wb_aft.fit(df, duration_col='week', event_col='arrest')
# print("✅ Fitting complete.")

# 5️⃣ Function to print summaries
def print_model_summaries():

    print("\n🔎 Log-Logistic AFT Summary:")
    ll_aft.print_summary()

    print("\n🔎 Log-Normal AFT Summary:")
    ln_aft.print_summary()

    print("\n🔎 Weibull AFT Summary:")
    wb_aft.print_summary()


# 6️⃣ Function to compare AIC scores
def print_model_comparisons():
    print(f"Log-Logistic AIC: {ll_aft.AIC_:.2f}")
    print(f"Log-Normal AIC: {ln_aft.AIC_:.2f}")
    print(f"Weibull AIC: {wb_aft.AIC_:.2f}")


# 7️⃣ Expose fitted models if needed
def get_fitted_models():
    return {
        "LogLogistic": ll_aft,
        "LogNormal": ln_aft,
        "Weibull": wb_aft
    }


def plot_survival_curves( sample_idx=0):
    """
    Plots survival curves for the 4 AFT models for a sample row.

    Args:
        sample_idx (int): Index of the row to use for the survival plot.
    """
    plt.figure(figsize=(10, 6))
    sample = df.iloc[[sample_idx]]

    sf_ll = ll_aft.predict_survival_function(sample)
    plt.plot(sf_ll.index, sf_ll.values.flatten(), label="Log-Logistic AFT")

    # Log-Normal
    sf_ln = ln_aft.predict_survival_function(sample)
    plt.plot(sf_ln.index, sf_ln.values.flatten(), label="Log-Normal AFT")

    # Weibull
    sf_wb = wb_aft.predict_survival_function(sample)
    plt.plot(sf_wb.index, sf_wb.values.flatten(), label="Weibull AFT")

    plt.title("Survival Functions Comparison")
    plt.xlabel("Time (weeks)")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.show()



def calculate_and_explore_clv(model, df, features, revenue_per_period=100, plot=True):
    """
    Calculate CLV per customer using the provided survival model and explore by segments.

    Args:
        model: A fitted lifelines AFT model (e.g., LogNormalAFTFitter).
        df (pd.DataFrame): The dataset with customer features.
        features (list): List of feature columns to use for prediction.
        revenue_per_period (float): Revenue per period (e.g., per week/month).
        plot (bool): Whether to show plots for segments.

    Returns:
        pd.DataFrame: The DataFrame with a new 'CLV' column.
        dict: Summary tables by key segments.
    """
    # 1️⃣ Predict median survival time
    predicted_median = model.predict_median(df[features])

    # 2️⃣ Calculate CLV
    df = df.copy()
    df['CLV'] = predicted_median * revenue_per_period

# Fallback if dummies exist
    def get_custcat(row):
        if row.get('custcat_E-service', 0) == 1:
            return 'E-service'
        elif row.get('custcat_Plus service', 0) == 1:
            return 'Plus service'
        elif row.get('custcat_Total service', 0) == 1:
            return 'Total service'
        else:
            return 'Other'

    df['Customer Category'] = df.apply(get_custcat, axis=1)
    if 'marital_Unmarried' in df.columns:
        df['Marital Status'] = df['marital_Unmarried'].map({1: 'Unmarried', 0: 'Married'})
    else:
        df['Marital Status'] = 'Unknown'

    # 4️⃣ Summaries
    summaries = {
    'Customer Category': df.groupby('Customer Category')['CLV'].agg(['count', 'mean', 'std', 'min', 'max']),
    'Marital Status': df.groupby('Marital Status')['CLV'].agg(['count', 'mean', 'std', 'min', 'max'])
}

    return df, summaries