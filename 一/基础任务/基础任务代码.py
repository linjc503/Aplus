# 数据处理
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.svm import SVC 
from sklearn.preprocessing import StandardScaler  

# 数据准备
train_data = pd.read_csv(r'C:\Users\12991\Desktop\小A\Task1\data\train.csv',index_col=None)
test_data = pd.read_csv(r'C:\Users\12991\Desktop\小A\Task1\data\test.csv',index_col=None)

# 训练集和测试集
x_train = train_data.iloc[:, 0:3]
y_train = train_data.iloc[:, 3]
x_test = test_data.iloc[:, 0:3]
y_test = test_data.iloc[:, 3]

print(f'训练集形状：{train_data.shape}\n')


# 1.绘制三维一刀切散点图
train_0 = train_data[train_data.iloc[:,3]==0]
train_1 = train_data[train_data.iloc[:,3]==1]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(train_0.iloc[:,0], train_0.iloc[:,1], train_0.iloc[:,2], c='b',label='标签0')
ax.scatter(train_1.iloc[:,0], train_1.iloc[:,1], train_1.iloc[:,2], c='r',label='标签1')

ax.set_xlabel('特征1')
ax.set_ylabel('特征2')
ax.set_zlabel('特征3')
ax.set_title('三维一刀切散点图')
ax.legend()
plt.show()


# 2.训练一个分类器
svm = SVC(kernel='linear',random_state=42)

svm.fit(x_train,y_train)


# 3.测试并统计准确率
y_pred = svm.predict(x_test)

acc = (y_pred==y_test).sum() / len(y_test)

print(f'准确率为{acc*100:.2f}%\n')  # 94.00%


# 4.标准化并比较

w = svm.coef_[0]
b = svm.intercept_[0]
print(f'超平面方程：{w[0]:.3f}x + {w[1]:.3f}y + {w[2]:.3f}z + {b:.3f}\n')
# 超平面方程：0.931x + 0.908y + 1.902z + -4.396

scaler = StandardScaler()

x_train_s = scaler.fit_transform(x_train)
x_test_s = scaler.transform(x_test)

svm_s = SVC(kernel='linear')

svm_s.fit(x_train_s,y_train)
y_pred_s = svm_s.predict(x_test_s)
acc_s = (y_pred_s==y_test).sum() / len(y_test)
print(f'标准化后准确率为{acc_s*100:.2f}%\n')  # 93.33%

w2 = svm_s.coef_[0]
b2 = svm_s.intercept_[0]
print(f'超平面方程：{w2[0]:.3f}x + {w2[1]:.3f}y + {w2[2]:.3f}z + {b2:.3f}\n')
# 超平面方程：1.327x + 1.327y + 0.601z + 0.024
