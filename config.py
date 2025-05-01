import pandas as pd

# Read data
df_telco = pd.read_csv('data/telco.csv')

def preprocess_data(df_telco: pd.DataFrame) -> pd.DataFrame:
    df = df_telco.copy()

    # 1️⃣ Binary encode Yes/No columns
    binary_cols = ['voice', 'internet', 'forward', 'retire', 'churn']
    df[binary_cols] = df[binary_cols].apply(lambda x: x.map({'Yes': 1, 'No': 0}))

    # 2️⃣ One-hot encode categorical columns (drop_first=True avoids dummy trap)
    categorical_cols = ['region', 'marital', 'gender', 'custcat', 'ed']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # ✅ Optionally: rename duration and event columns for clarity (if needed)
    # E.g., if your AFT script uses 'week' and 'arrest', but your real data is 'tenure' and 'churn'
    # You can do this mapping to align:
    df = df.rename(columns={'tenure': 'week', 'churn': 'arrest'})

    df['week'] = df['week'] / 10

    return df

# This is the DataFrame your main script imports
df = preprocess_data(df_telco)

# ✅ Debug prints (can be removed in production)

if __name__ == "__main__":
    # 1️⃣ Scale the duration
    print(df.isnull().sum())
    print((df == 0).sum())
