import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# 1. Load the shared trading file
data_file = Path(__file__).resolve().parent / "trades_orderbook_clean_ANSWER_KEY.csv"
if not data_file.is_file():
    raise FileNotFoundError(
        f"Required input file not found: {data_file}\n"
        "Place trades_orderbook_clean_ANSWER_KEY.csv in the same folder as "
        "eligibility.py, or change data_file to the CSV's actual location."
    )

df = pd.read_csv(data_file)
print("Trading data loaded:", df.shape)

# IMPORTANT:
# The trading file has no credit score or loan approval columns.
# These are SIMULATED loan data for assessment demonstration.
np.random.seed(42)
n = len(df)

loan = pd.DataFrame({
    "credit_score": np.random.randint(300, 901, n),
    "loan_amount": np.random.randint(50000, 1000001, n)
})

# Simulated eligibility rule for demonstration only
loan["eligible"] = (
    (loan["credit_score"] >= 650) &
    (loan["loan_amount"] <=
     loan["credit_score"] * 1000)
).astype(int)

# 2. Select input and output
X = loan[["credit_score", "loan_amount"]]
y = loan["eligible"]

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y
)

# 4. Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 5. Predict eligibility
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

metrics_df = pd.DataFrame({"metric": ["accuracy"], "value": [accuracy]})
metrics_df.to_csv(output_dir / "model_metrics.csv", index=False)

# 6. Predict for a new applicant
score = float(input("Enter credit score (300-900): "))
amount = float(input("Enter loan amount required: "))

applicant = pd.DataFrame({
    "credit_score": [score],
    "loan_amount": [amount]
})

result = model.predict(applicant)[0]
result_text = "Loan Eligible" if result == 1 else "Loan Not Eligible"
print(result_text)

applicant_result = pd.DataFrame({
    "credit_score": [score],
    "loan_amount": [amount],
    "predicted_eligibility": [result],
    "status": [result_text]
})
applicant_result.to_csv(output_dir / "applicant_prediction.csv", index=False)

# 7. Plot credit score vs loan amount
colors = loan["eligible"].map({1: "green", 0: "red"})

plt.figure(figsize=(9, 6))
plt.scatter(
    loan["credit_score"], loan["loan_amount"],
    c=colors, alpha=0.6
)

plt.xlabel("Credit Score (CIBIL Score)")
plt.ylabel("Loan Amount Required (₹)")
plt.title("Loan Eligibility using Logistic Regression")
plt.grid(True, alpha=0.3)

plt.scatter([], [], color="green", label="Eligible")
plt.scatter([], [], color="red", label="Not Eligible")
plt.legend()
plt.tight_layout()
plot_path = output_dir / "loan_eligibility_plot.png"
plt.savefig(plot_path, dpi=200)
plt.close()

print(f"Outputs saved to: {output_dir}")
