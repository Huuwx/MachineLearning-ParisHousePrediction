import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import pickle;
from scikeras.wrappers import KerasRegressor

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


scaler = StandardScaler()

X_Train_scaled = scaler.fit_transform(X_Train)
X_Test_scaled = scaler.transform(X_Test)

y = np.array(Y_Test)

#open LassoModel from file
with open('LassoModel', 'rb')as file:
    lasso = pickle.load(file);
    
#open LinearModel from file
with open('LinearModel', 'rb')as file:
    linear = pickle.load(file);
 
#open NeuralNetworkModel from file   
with open('NeuralNetworkModel', 'rb')as file:
    model = pickle.load(file);
    
ann = KerasRegressor(model= model)
    
estimator_list = [
    ('lasso', lasso),
    ('linear', linear),
    ('ann', ann)
]

#Build stack model
stack_model = StackingRegressor(
    estimators = estimator_list, final_estimator = LinearRegression()
)


#Train stacked model
stack_model.fit(X_Train_scaled, Y_Train)

file_name = 'StackingModel'
with open(file_name, 'wb') as file:
    pickle.dump(stack_model, file)

#Make predictions
y_pred= stack_model.predict(X_Test_scaled)

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

print("Đánh giá mô hình Stacking bằng độ đo R2 %.12f" %r2_score(Y_Test, y_pred))
print("Đánh giá mô hình Stacking bằng độ đo MAE: %.12f" % mean_absolute_error(Y_Test, y_pred))
print("Đánh giá mô hình Stacking bằng độ đo RMSE: %.12f" % np.sqrt(mean_squared_error(Y_Test, y_pred)))

# Nhận xét:
if r2_score(Y_Test, y_pred) > 0.7:
    print("Mô hình có độ chính xác khá cao.")
elif r2_score(Y_Test, y_pred) > 0.5:
    print("Mô hình có độ chính xác trung bình.")
else:
    print("Mô hình có độ chính xác thấp.")