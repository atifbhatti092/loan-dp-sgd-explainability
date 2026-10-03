"""Leak-free preprocessing: every statistic (mode, median, scaler) is fitted on the TRAINING fold only."""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

def load_raw(path="../Train.csv"):
    df = pd.read_csv(path).drop(columns=["Loan_ID"])
    y = df.pop("Loan_Status").map({"Y": 1, "N": 0}).values.astype(float)
    return df, y

CAT = ["Gender", "Married", "Dependents", "Self_Employed"]
LOGS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "TotalIncome", "EMI", "Balance_Income"]

class Prep:
    """credit_mode: 'mode' (paper), 'indicator' (mode + missing flag), 'category' (own state for missing)"""
    def __init__(self, credit_mode="mode", drop_protected=False):
        self.credit_mode, self.drop_protected = credit_mode, drop_protected
    def _clean(self, d):
        d = d.copy()
        for c in CAT: d[c] = d[c].fillna(self.modes[c])
        d["LoanAmount"] = d["LoanAmount"].fillna(self.med["LoanAmount"])
        d["Loan_Amount_Term"] = d["Loan_Amount_Term"].fillna(self.med["Loan_Amount_Term"])
        miss = d["Credit_History"].isna().astype(float)
        d["Credit_History"] = d["Credit_History"].fillna(self.modes["Credit_History"])
        d["Dependents"] = d["Dependents"].replace("3+", 3).astype(float)
        for c, m in [("Gender", {"Male": 1, "Female": 0}), ("Married", {"Yes": 1, "No": 0}),
                     ("Self_Employed", {"Yes": 1, "No": 0})]: d[c] = d[c].map(m)
        d["Education"] = d["Education"].map({"Graduate": 1, "Not Graduate": 0})
        for a in ["Rural", "Semiurban", "Urban"]: d["PA_" + a] = (d["Property_Area"] == a).astype(float)  # one-hot, no fake order
        d["TotalIncome"] = d["ApplicantIncome"] + d["CoapplicantIncome"]
        d["EMI"] = d["LoanAmount"]*1000.0/d["Loan_Amount_Term"]
        d["Income_to_Loan_Ratio"] = d["TotalIncome"]/(d["LoanAmount"]*1000.0)
        d["Balance_Income"] = d["TotalIncome"] - d["EMI"]
        for c in LOGS: d[c + "_log"] = np.sign(d[c])*np.log1p(np.abs(d[c]))
        cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Loan_Amount_Term", "Credit_History",
                "PA_Rural", "PA_Semiurban", "PA_Urban", "Income_to_Loan_Ratio"] + [c + "_log" for c in LOGS]
        if self.drop_protected: cols = [c for c in cols if c not in ("Gender", "Married", "Dependents")]
        if self.credit_mode == "indicator": d["CH_missing"] = miss; cols.append("CH_missing")
        if self.credit_mode == "category":
            d["CH_missing"] = miss; d.loc[miss == 1, "Credit_History"] = 0.5; cols.append("CH_missing")
        self.raw_credit = d["Credit_History"].values.copy()
        return d[cols], cols
    def fit(self, d):
        self.modes = {c: d[c].mode()[0] for c in CAT + ["Credit_History"]}
        self.med = {c: d[c].median() for c in ["LoanAmount", "Loan_Amount_Term"]}
        Z, self.cols = self._clean(d)
        self.train_credit = self.raw_credit
        self.scaler = StandardScaler().fit(Z.values.astype(float))
        return self
    def transform(self, d):
        Z, _ = self._clean(d)
        return self.scaler.transform(Z.values.astype(float))
    def fit_transform(self, d): self.fit(d); return self.transform(d)
