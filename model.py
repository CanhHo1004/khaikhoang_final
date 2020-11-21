# Khai báo thư viện cần thiết
import pandas as pd
from sklearn import preprocessing
from scipy.io import arff
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import pickle

# Nạp dữ liệu vào
df = pd.read_csv('https://raw.githubusercontent.com/CanhHo1004/dataset/main/divorce/divorce.csv',';')
# print(df)

# Kiểm tra giá trị null trong tập dữ liệu
total =  df.isnull().sum()
# print(total)

# Lấy các cột thuôc tính
X = df.drop('Class', axis=1)
# Lấy cột nhãn
y = df['Class']

# Hàm xây dựng mô hình sử dụng K-FOld
def score_dt(X, y):
  kf = KFold(n_splits= 10)                    # Chọn số fold = 10
  total = 0
  for train_index, test_index in kf.split(X): # Tiến hành phân chia
    X_train, X_test = X.iloc[train_index], X.iloc[test_index] # Lấy X_train, X_test theo từng phần được chia của X
    y_train, y_test = y.iloc[train_index], y.iloc[test_index] # Lấy y_train, y_test theo từng phần được chia của y
    model = DecisionTreeClassifier()          # Khai báo mô hình sử dụng
    model.fit(X_train, y_train)               # Tiến hành huấn luyện
    y_pred = model.predict(X_test)            # Tiến hành dự đoán
    total += accuracy_score(y_test, y_pred)   # Tổng độ chính xác qua các phần
  pickle.dump(model, open('model.pkl','wb'))  # Xuất ra file model.pkl
  return total/10                             # Trả về độ chính xác trung bình

test = score_dt(X, y)                         # Chạy thử hàm score_dt với X, y là 2 giá trị được lấy ở phía trên
print(test)
