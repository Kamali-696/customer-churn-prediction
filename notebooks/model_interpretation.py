# Checking the feature importance of the features for model interpretation and EDA comparision
from main import best_lr
lr_model = best_lr.named_steps["model"]

coef = lr_model.coef_[0]

importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": coef
})

importance_df = importance_df.sort_values(
    by="Coefficient",
    ascending=False
)

# higher the values - more likely to be churned
print(importance_df.head(10))

print()
print()

# higher the values - less likely to be churned
print(importance_df.tail(10))

