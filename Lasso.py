import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import pickle;

data = pd.read_csv('ParisHousingClass.csv')

# Thay thế các giá trị NaN
data.replace(['NaN', 'N/A', 'NA', 'n/a', 'n.a.', 'N#A', 'n#a', '?'], 'other', inplace=True)

# Phan loai bien
numerical_vars = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_vars = data.select_dtypes(include=['object']).columns.tolist()

print('Numerical variables:', numerical_vars)
print('Categorical variables:', categorical_vars)


# Check if 'category' column exists before mapping
if 'category' in data.columns:
    # Map 'category' column: 'Basic' -> 0, 'Luxury' -> 1
    data['category'] = data['category'].map({'Basic': 0, 'Luxury': 1})
else:
    print("The 'category' column does not exist in the dataset.")


dt_Train, dt_Test = train_test_split(data, test_size=0.3, shuffle=False)

#Displaying the first few rows
print(data.head())

drop_columns = ['price']

X_Train = dt_Train.drop(drop_columns, axis = 1)
Y_Train = dt_Train['price']
X_Test = dt_Test.drop(drop_columns, axis = 1)
Y_Test = dt_Test['price']

# Chuẩn hóa dữ liệu
scaler = StandardScaler()

X_Train_scaled = scaler.fit_transform(X_Train)
X_Test_scaled = scaler.transform(X_Test)

# Lưu scaler vào file
scaler_file = 'scaler'
with open(scaler_file, 'wb') as file:
    pickle.dump(scaler, file)

y = np.array(Y_Test)

# model = Lasso(alpha=0.1).fit(X_Train_scaled, Y_Train)

# Khởi tạo mô hình Lasso
lasso = Lasso()

# Xác định danh sách các giá trị alpha cần thử nghiệm
alpha_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

# Thiết lập GridSearchCV
param_grid = {'alpha': alpha_values}
grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_Train_scaled, Y_Train)

# Lấy giá trị alpha tốt nhất
best_alpha = grid_search.best_params_['alpha']
print("Best alpha for Lasso: ", best_alpha)

# Khởi tạo lại mô hình Lasso với alpha tốt nhất
model = Lasso(alpha=best_alpha)
model.fit(X_Train_scaled, Y_Train)

#save model to file
file_name = 'LassoModel'
with open(file_name, 'wb') as file:
    pickle.dump(model, file)
#open model from file
# with open(file_name, 'rb')as file:
#     model2 = pickle.load(file);

y_pred= model.predict(X_Test_scaled)

print("Thuc te        Du doan      Chenh lech")
for i in range(0,10):
    print("%.2f" % y[i],"   ",  "%.2f" % y_pred[i], "   ", abs(y[i]-y_pred[i]))

# 1. Actual vs Predicted Plot (Testing Set)
plt.figure(figsize=(10, 5))
plt.scatter(Y_Test, y_pred, alpha=0.6)
plt.plot([Y_Test.min(), Y_Test.max()], [Y_Test.min(), Y_Test.max()], '--r', linewidth=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values (Test Set)")
plt.show()

# 2. Residual Plot (Testing Set)
plt.figure(figsize=(10, 5))
residuals = Y_Test - y_pred
sns.histplot(residuals, kde=True)
plt.xlabel("Residuals")
plt.title("Residual Distribution (Test Set)")
plt.show()

# 3. Predicted vs Residuals Plot (Testing Set)
plt.figure(figsize=(10, 5))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Predicted vs Residuals (Test Set)")
plt.show()

print("Đánh giá mô hình Lasso bằng độ đo R2 %.12f" %r2_score(Y_Test, y_pred))
print("Đánh giá mô hình Lasso bằng độ đo MAE: %.12f" % mean_absolute_error(Y_Test, y_pred))
print("Đánh giá mô hình Lasso bằng độ đo RMSE: %.12f" % np.sqrt(mean_squared_error(Y_Test, y_pred)))

# Nhận xét:
if r2_score(Y_Test, y_pred) > 0.7:
    print("Mô hình có độ chính xác khá cao.")
elif r2_score(Y_Test, y_pred) > 0.5:
    print("Mô hình có độ chính xác trung bình.")
else:
    print("Mô hình có độ chính xác thấp.")