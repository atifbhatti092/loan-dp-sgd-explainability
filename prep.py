import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RNG = 42
np.random.seed(RNG)

df = pd.read_csv("Train.csv")
df = df.drop(columns=["Loan_ID"])

# ---- Missing value imputation ----
for c in ["Gender", "Married", "Dependents", "Self_Employed"]:
    df[c] = df[c].fillna(df[c].mode()[0])
df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median())
df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())

# ---- Encode ----
df["Dependents"] = df["Dependents"].replace("3+", 3).astype(int)
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Married"] = df["Married"].map({"Yes": 1, "No": 0})
df["Education"] = df["Education"].map({"Graduate": 1, "Not Graduate": 0})
df["Self_Employed"] = df["Self_Employed"].map({"Yes": 1, "No": 0})
df["Property_Area"] = df["Property_Area"].map({"Rural": 0, "Semiurban": 1, "Urban": 2})
y = df["Loan_Status"].map({"Y": 1, "N": 0}).values
df = df.drop(columns=["Loan_Status"])

# ---- Feature engineering ----
df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
# LoanAmount is expressed in thousands in this dataset; EMI approximated as principal/term (no-interest annuity proxy)
df["EMI"] = (df["LoanAmount"] * 1000.0) / df["Loan_Amount_Term"]
df["Income_to_Loan_Ratio"] = df["TotalIncome"] / (df["LoanAmount"] * 1000.0)
df["Balance_Income"] = df["TotalIncome"] - df["EMI"]

# Log-transform skewed monetary fields (log1p keeps zeros defined)
for c in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "TotalIncome", "EMI", "Balance_Income"]:
    df[c + "_log"] = np.sign(df[c]) * np.log1p(np.abs(df[c]))

feature_cols = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "Loan_Amount_Term", "Credit_History", "Property_Area",
    "Income_to_Loan_Ratio",
    "ApplicantIncome_log", "CoapplicantIncome_log", "LoanAmount_log",
    "TotalIncome_log", "EMI_log", "Balance_Income_log",
]
X = df[feature_cols].values.astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RNG
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print("Train:", X_train_s.shape, "Test:", X_test_s.shape)
print("Train class balance:", np.bincount(y_train))
print("Test class balance:", np.bincount(y_test))

np.savez("prep_data.npz", X_train=X_train_s, X_test=X_test_s,
         y_train=y_train, y_test=y_test, feature_cols=np.array(feature_cols))
print("Saved prep_data.npz")
