import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import log_loss, roc_auc_score, roc_curve, confusion_matrix


# Colours
colors = ["#1F4E79", "#2A9D8F", "#E9C46A", "#E76F51"]
out_pal = ["#E76F51", "#2A9D8F"]


# 1. LOAD DATASET

df = pd.read_csv("himalayan_joined.csv")

print(df.head(10))
print(df.info())
print("Rows and Columns:", df.shape)
print(df.describe())

print(df.isna().sum())


# 2. EDA

sns.countplot(x="success", data=df, hue="success",
              palette=out_pal, legend=False)
plt.xticks([0, 1], ["Failed", "Successful"])
plt.title("Expedition Outcome")
plt.show()


sns.countplot(x="season", data=df, hue="season",
              palette=colors, legend=False)
plt.title("Expeditions by Season")
plt.show()





# 3. HANDLING MISSING VALUES

df["duration_days"] = pd.to_numeric(df["duration_days"], errors="coerce")

df["duration_days"] = df["duration_days"].fillna(
    df["duration_days"].median()
)

df = df.dropna(subset=["duration_days"])
df = df.drop_duplicates().reset_index(drop=True)
print(df.isna().sum())



# 4. CREATE HEIGHT CATEGORY

df.loc[df["height_metres"] < 7500, "height_category"] = "Below 7500m"

df.loc[(df["height_metres"] >= 7500) &
       (df["height_metres"] < 8000), "height_category"] = "7500m-7999m"

df.loc[df["height_metres"] >= 8000, "height_category"] = "8000m and above"


# 5. CORRELATION ANALYSIS

corr_cols = ["height_metres", "year", "members", "hired_staff",
             "oxygen_used", "agency_used", "duration_days", "success"]

print(df[corr_cols].corr())

sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f",
            cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# Success by Oxygen

sns.barplot(x="oxygen_used", y="success", data=df,
            hue="oxygen_used", palette=colors[:2],
            errorbar=None, legend=False)

plt.title("Success Rate by Oxygen Use")
plt.show()


# Success by Season

sns.barplot(x="season", y="success", data=df,
            hue="season", palette=colors,
            errorbar=None, legend=False)

plt.title("Success Rate by Season")
plt.show()


# Success by Height Category

sns.barplot(x="height_category", y="success", data=df,
            hue="height_category", palette=colors[:3],
            errorbar=None, legend=False)

plt.title("Success Rate by Height Category")
plt.show()


# Success over Years

yearly = df.groupby("year")["success"].mean()

plt.plot(yearly.index, yearly.values, color=colors[0])
plt.title("Success Rate Over the Years")
plt.xlabel("Year")
plt.ylabel("Success Rate")
plt.grid(True)
plt.show()


# Height and Duration

sns.scatterplot(x="height_metres", y="duration_days",
                data=df, color=colors[1], alpha=0.3)

plt.title("Height vs Duration")
plt.show()


# 6. ENCODING AND NORMALIZATION

df = pd.get_dummies(df, columns=["season", "height_category"], dtype=int)

scale_cols = ["height_metres", "year", "members", "hired_staff"]

scaler = MinMaxScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

print(df[scale_cols].describe())

data = df.drop(columns=["peak_name"])

print(data.head())


# 7. SIMPLE LINEAR REGRESSION

X = data[["height_metres"]]
y = data["duration_days"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

slr = LinearRegression()
slr.fit(X_train, y_train)

y_pred = slr.predict(X_test)

slr_mae = mean_absolute_error(y_test, y_pred)
slr_mse = mean_squared_error(y_test, y_pred)
slr_rmse = np.sqrt(slr_mse)
slr_r2 = r2_score(y_test, y_pred)

print("\nSimple Linear Regression")
print("MAE:", slr_mae)
print("MSE:", slr_mse)
print("RMSE:", slr_rmse)
print("R2:", slr_r2)


plt.scatter(X_test, y_test, color=colors[0], label="Actual")
plt.scatter(X_test, y_pred, color=colors[3], label="Predicted")

plt.title("Simple Linear Regression")
plt.xlabel("Height")
plt.ylabel("Duration")
plt.legend()
plt.show()


# 8. MULTIPLE LINEAR REGRESSION

X = data[["height_metres", "year", "members", "hired_staff",
          "oxygen_used", "agency_used"]]

y = data["duration_days"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlr = LinearRegression()
mlr.fit(X_train, y_train)

y_pred = mlr.predict(X_test)

mlr_mae = mean_absolute_error(y_test, y_pred)
mlr_mse = mean_squared_error(y_test, y_pred)
mlr_rmse = np.sqrt(mlr_mse)
mlr_r2 = r2_score(y_test, y_pred)

print("\nMultiple Linear Regression")
print("MAE:", mlr_mae)
print("MSE:", mlr_mse)
print("RMSE:", mlr_rmse)
print("R2:", mlr_r2)


plt.scatter(y_test, y_pred, color=colors[0])
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], color=colors[3])

plt.title("Actual vs Predicted Duration")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()


# 9. POLYNOMIAL REGRESSION

X = data[["height_metres"]]
y = data["duration_days"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Degree 2

poly2 = PolynomialFeatures(degree=2)

X_train_poly2 = poly2.fit_transform(X_train)
X_test_poly2 = poly2.transform(X_test)

poly_model2 = LinearRegression()
poly_model2.fit(X_train_poly2, y_train)

y_pred = poly_model2.predict(X_test_poly2)

print("\nPolynomial Regression Degree 2")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))


X_sorted = np.sort(X_test, axis=0)
X_sorted_poly = poly2.transform(X_sorted)

plt.scatter(X_test, y_test, color=colors[1], alpha=0.3)
plt.plot(X_sorted, poly_model2.predict(X_sorted_poly),
         color=colors[3])

plt.title("Polynomial Regression Degree 2")
plt.xlabel("Height")
plt.ylabel("Duration")
plt.show()


# Degree 3

poly3 = PolynomialFeatures(degree=3)

X_train_poly3 = poly3.fit_transform(X_train)
X_test_poly3 = poly3.transform(X_test)

poly_model3 = LinearRegression()
poly_model3.fit(X_train_poly3, y_train)

y_pred = poly_model3.predict(X_test_poly3)

print("\nPolynomial Regression Degree 3")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))


X_sorted_poly = poly3.transform(X_sorted)

plt.scatter(X_test, y_test, color=colors[1], alpha=0.3)
plt.plot(X_sorted, poly_model3.predict(X_sorted_poly),
         color=colors[3])

plt.title("Polynomial Regression Degree 3")
plt.xlabel("Height")
plt.ylabel("Duration")
plt.show()


# 10. CLASSIFICATION

X = data.drop(columns=["success", "duration_days"])
y = data["success"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

labels = ["Failed", "Successful"]


# LOGISTIC REGRESSION

log_model = LogisticRegression()
log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

print("\nLogistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Log Loss:", log_loss(y_test, y_prob))
print("AUC:", roc_auc_score(y_test, y_prob))
print(confusion_matrix(y_test, y_pred))


sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr, color=colors[0], label="Logistic Regression")
plt.plot([0, 1], [0, 1], "k--")
plt.title("Logistic Regression ROC Curve")
plt.legend()
plt.show()


# KNN

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
y_prob = knn.predict_proba(X_test)[:, 1]

print("\nKNN")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Log Loss:", log_loss(y_test, y_prob))
print("AUC:", roc_auc_score(y_test, y_prob))
print(confusion_matrix(y_test, y_pred))


sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt="d", cmap="Blues")
plt.title("KNN Confusion Matrix")
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr, color=colors[1], label="KNN")
plt.plot([0, 1], [0, 1], "k--")
plt.title("KNN ROC Curve")
plt.legend()
plt.show()


# NAIVE BAYES

nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)
y_prob = nb.predict_proba(X_test)[:, 1]

print("\nNaive Bayes")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Log Loss:", log_loss(y_test, y_prob))
print("AUC:", roc_auc_score(y_test, y_prob))
print(confusion_matrix(y_test, y_pred))


sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt="d", cmap="Blues")
plt.title("Naive Bayes Confusion Matrix")
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr, color=colors[2], label="Naive Bayes")
plt.plot([0, 1], [0, 1], "k--")
plt.title("Naive Bayes ROC Curve")
plt.legend()
plt.show()


# DECISION TREE

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)
y_prob = dt.predict_proba(X_test)[:, 1]

print("\nDecision Tree")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Log Loss:", log_loss(y_test, y_prob))
print("AUC:", roc_auc_score(y_test, y_prob))
print(confusion_matrix(y_test, y_pred))


sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt="d", cmap="Blues")
plt.title("Decision Tree Confusion Matrix")
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr, color=colors[3], label="Decision Tree")
plt.plot([0, 1], [0, 1], "k--")
plt.title("Decision Tree ROC Curve")
plt.legend()
plt.show()


# Decision Tree Diagram

plt.figure(figsize=(16, 8))

plot_tree(dt, max_depth=3, feature_names=X.columns,
          class_names=labels, filled=True)

plt.title("Decision Tree Structure")
plt.show()




# SUPPORT VECTOR MACHINE

svm = SVC(kernel="rbf", probability=True, random_state=42)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
y_prob = svm.predict_proba(X_test)[:, 1]

print("\nSupport Vector Machine")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Log Loss:", log_loss(y_test, y_prob))
print("AUC:", roc_auc_score(y_test, y_prob))
print(confusion_matrix(y_test, y_pred))


sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True, fmt="d", cmap="Blues")
plt.title("SVM Confusion Matrix")
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr, color=colors[0], label="SVM")
plt.plot([0, 1], [0, 1], "k--")
plt.title("SVM ROC Curve")
plt.legend()
plt.show()
