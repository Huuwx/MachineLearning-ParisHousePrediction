import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle;

data = pd.read_csv('ParisHousing.csv')

#Displaying the first few rows
print(data.head())

dt_Train, dt_Test = train_test_split(data, test_size=0.2, shuffle=False)

X_Train = dt_Train.iloc[:, 0:16]
Y_Train = dt_Train.iloc[:, -1]
X_Test = dt_Test.iloc[:, 0:16]
Y_Test = dt_Test.iloc[:, -1]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()

X_Train_scaled = scaler.fit_transform(X_Train)
X_Test_scaled = scaler.transform(X_Test)

# Lưu scaler vào file
scaler_file = 'scaler'
with open(scaler_file, 'wb') as file:
    pickle.dump(scaler, file)

y = np.array(Y_Test)

model = Lasso().fit(X_Train_scaled, Y_Train)

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

def tinh_nse(y_test, y_pred):
    sse = np.sum((y_test - y_pred) ** 2)
    sst = np.sum((y_test - np.mean(y_test)) ** 2)
    nse = 1 - (sse / sst)
    return nse

# Biểu đồ giá thực tế so với giá dự đoán
plt.scatter(Y_Test, y_pred, color='blue')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs. Predicted Prices (Linear Regression)')
plt.show()

print("Đánh giá mô hình Lasso bằng độ đo R2 %.12f" %r2_score(Y_Test, y_pred))
print("Đánh giá mô hình Lasso bằng độ đo NSE: %.12f" % tinh_nse(Y_Test, y_pred))
print("Đánh giá mô hình Lasso bằng độ đo MAE: %.12f" % mean_absolute_error(Y_Test, y_pred))
print("Đánh giá mô hình Lasso bằng độ đo RMSE: %.12f" % np.sqrt(mean_squared_error(Y_Test, y_pred)))

# Nhận xét:
if r2_score(Y_Test, y_pred) > 0.7:
    print("Mô hình có độ chính xác khá cao.")
elif r2_score(Y_Test, y_pred) > 0.5:
    print("Mô hình có độ chính xác trung bình.")
else:
    print("Mô hình có độ chính xác thấp.")