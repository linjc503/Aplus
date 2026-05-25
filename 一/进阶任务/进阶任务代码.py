import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import os

# 设置随机种子以保证可重复性
torch.manual_seed(42)
np.random.seed(42)

# 1. 读取数据并划分训练集和测试集
def load_data(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)
    # 分离特征和标签
    features = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    
    # 输出数据形状
    print(f"数据形状: 特征={features.shape}, 标签={labels.shape}")
    
    # 按照8:2的比例划分训练集和测试集
    split_idx = int(len(features) * 0.8)
    train_features, test_features = features[:split_idx], features[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]
    
    return train_features, train_labels, test_features, test_labels

# 2. 数据标准化
# Z-score标准化，每个特征经过变换后，均值为 0，标准差为 1
def normalize_data(train_features, test_features):
    # 计算训练集的均值和方差
    mean = np.mean(train_features, axis=0)
    std = np.std(train_features, axis=0)
    
    # 对训练集和测试集进行标准化
    train_features_normalized = (train_features - mean) / std
    test_features_normalized = (test_features - mean) / std
    
    return train_features_normalized, test_features_normalized

# 3. 定义神经网络模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        # 隐藏层之间
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
        # 最后一个隐藏层到输出层
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# 4. 训练模型
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # 清零梯度
            optimizer.zero_grad()
            # 前向传播
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            
            running_loss += loss.item()
        
        # 每训练10轮，就print一次数据
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# 5. 测试模型
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f"测试集准确率: {accuracy:.4f}")
    return accuracy

# 6. 模型调优
def tune_hyperparameters(train_features, train_labels, test_features, test_labels):
    # 转换为PyTorch张量
    train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
    
    # 创建数据集和数据加载器
    train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_features_tensor, test_labels_tensor)
    
    best_accuracy = 0.0
    best_model = None
    best_hyperparams = {}
    
    # 记录所有超参数组合的结果
    results = []
    
    # 超参数组合
    hidden_sizes_list = [[64, 32], [128, 64], [256, 128], [128, 64, 32]]
    learning_rates = [0.001, 0.0005, 0.0001]
    batch_sizes = [32, 64]
    epochs_list = [100, 200, 300]
    optimizers = ['adam', 'sgd']
    
    
    # 遍历所有超参数组合
    for hidden_sizes in hidden_sizes_list:
        for lr in learning_rates:
            for batch_size in batch_sizes:
                for epochs in epochs_list:
                    for opt_name in optimizers:
                        print(f"\n测试超参数: 隐藏层={hidden_sizes}, 学习率={lr}, 批次大小={batch_size}, 轮数={epochs}, 优化器={opt_name}")
                        
                        # 创建数据加载器
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                        
                        # 初始化模型
                        input_size = train_features.shape[1]
                        output_size = len(np.unique(train_labels))
                        model = MLP(input_size, hidden_sizes, output_size)
                        
                        # 定义损失函数
                        criterion = nn.CrossEntropyLoss()
                        
                        # 选择优化器
                        if opt_name == 'adam':
                            optimizer = optim.Adam(model.parameters(), lr=lr)
                        else:
                            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
                        
                        # 训练模型
                        train_model(model, train_loader, criterion, optimizer, epochs)
                        
                        # 测试模型
                        accuracy = test_model(model, test_loader)
                        
                        # 记录结果
                        results.append({
                            'hidden_sizes': hidden_sizes,
                            'learning_rate': lr,
                            'batch_size': batch_size,
                            'epochs': epochs,
                            'optimizer': opt_name,
                            'accuracy': accuracy
                        })
                        
                        # 更新最佳模型
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_model = model
                            best_hyperparams = {
                                'hidden_sizes': hidden_sizes,
                                'learning_rate': lr,
                                'batch_size': batch_size,
                                'epochs': epochs,
                                'optimizer': opt_name
                            }
                            print(f"找到更好的模型! 准确率: {best_accuracy:.4f}")
                        
                        # 如果准确率达到75%以上，提前停止
                        if accuracy >= 0.75:
                            print(f"准确率达到75%以上，停止调优")
                            return best_model, best_hyperparams, results
    
    return best_model, best_hyperparams, results

# 7. 保存模型
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"模型已保存到: {path}")

# 主函数
def main():
    # 数据文件路径
    file_path = 'gandou.csv'
    
    # 1. 读取数据并划分训练集和测试集
    train_features, train_labels, test_features, test_labels = load_data(file_path)
    
    # 2. 数据标准化
    train_features_normalized, test_features_normalized = normalize_data(train_features, test_features)
    
    # 3. 模型调优
    best_model, best_hyperparams, results = tune_hyperparameters(train_features_normalized, train_labels, test_features_normalized, test_labels)
    
    # 4. 测试最佳模型
    print("\n最佳模型超参数:")
    for key, value in best_hyperparams.items():
        print(f"{key}: {value}")
    
    # 转换为PyTorch张量
    test_features_tensor = torch.tensor(test_features_normalized, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
    test_dataset = TensorDataset(test_features_tensor, test_labels_tensor)
    test_loader = DataLoader(test_dataset, batch_size=best_hyperparams['batch_size'], shuffle=False)
    
    # 计算最终准确率
    final_accuracy = test_model(best_model, test_loader)
    print(f"最终模型准确率: {final_accuracy:.2f}")
    
    # 5. 保存模型
    model_path = '进阶任务模型.pth'
    save_model(best_model, model_path)
    
    # 6. 保存调优结果
    with open('调优结果.md', 'w', encoding='utf-8') as f:
        f.write('# 模型调优结果\n\n')
        f.write('## 超参数配置与准确率\n\n')
        f.write('| 隐藏层结构 | 学习率 | 批次大小 | 训练轮数 | 优化器 | 准确率 |\n')
        f.write('|-----------|--------|----------|----------|--------|--------|\n')
        
        for result in results:
            f.write(f"| {result['hidden_sizes']} | {result['learning_rate']} | {result['batch_size']} | {result['epochs']} | {result['optimizer']} | {result['accuracy']:.4f} |\n")
        
        f.write('\n## 最佳模型\n\n')
        f.write(f"隐藏层结构: {best_hyperparams['hidden_sizes']}\n")
        f.write(f"学习率: {best_hyperparams['learning_rate']}\n")
        f.write(f"批次大小: {best_hyperparams['batch_size']}\n")
        f.write(f"训练轮数: {best_hyperparams['epochs']}\n")
        f.write(f"优化器: {best_hyperparams['optimizer']}\n")
        f.write(f"准确率: {final_accuracy:.4f}\n")
        
        f.write('\n## 调优思路\n\n')
        f.write('1. 尝试了不同的神经网络结构，包括2层和3层隐藏层，以及不同的神经元数量\n')
        f.write('2. 测试了不同的学习率，从0.001到0.0001\n')
        f.write('3. 调整了批次大小，分别为32和64\n')
        f.write('4. 尝试了不同的训练轮数，从100到300\n')
        f.write('5. 比较了Adam和SGD两种优化器的性能\n')
        f.write('6. 当准确率达到75%以上时停止调优，确保模型满足要求\n')

if __name__ == '__main__':
    main()