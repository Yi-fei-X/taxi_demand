import numpy as np
import xgboost as xgb
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
import csv
from prettytable import PrettyTable
from sklearn.metrics import mean_squared_error

torch.manual_seed(1)

# Assuming that we are on a CUDA machine, this should print a CUDA device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#Load Data
data = np.load('train.npz')
x = data['x'] #feature matrix
y = data['y'] #label matrix
location = data['locations'] #location matrix
times = data['times'] #time matrix

data_val = np.load('val.npz')
x_val = data_val['x'] #feature matrix
y_val = data_val['y'] #label matrix
location_val = data_val['locations'] #location matrix
times_val = data_val['times'] #time matrix

data_test = np.load('test.npz')
x_test = data_test['x'] #feature matrix
location_test = data_test['locations'] #location matrix
times_test = data_test['times'] #time matrix

#RMSE
def RMSE(predict, target):
    return np.sqrt(mean_squared_error(target, predict))
    # return np.sqrt(((predict - target) ** 2).mean())

#Q1
n = len(x)
historical = np.zeros((10,10))
t_sum = len(location)/100

for i in range(n):
    j = location[i, 0]
    k = location[i, 1]
    historical[j, k] = historical[j, k] + y[i]
historical = historical/t_sum

n_val = len(x_val)
y_pred_his = np.zeros(n_val)
for i in range(n_val):
    j = location_val[i, 0]
    k = location_val[i, 1]
    y_pred_his[i] = historical[j, k]

RMSE_Q1 = RMSE(y_pred_his, np.squeeze(y_val))
print('RMSE_Q1: %.10f' % RMSE_Q1 )


#Extract features from temporal data
input_X = []
for i in range(n):
    input_X.append(x[i, :, 24])
input_X = np.array(input_X)

input_val = []
for i in range(n_val):
    input_val.append(x_val[i, :, 24])
input_val = np.array(input_val)

#Q2 Linear Regression
reg = LinearRegression().fit(input_X,y)
y_pred_reg = reg.predict(input_val)
RMSE_reg = RMSE(y_pred_reg, y_val)
print('RMSE_reg: %.10f' % RMSE_reg)

#Q3 XGBOOST
dtc = xgb.XGBClassifier()
dtc.fit(input_X, np.squeeze(y))
y_pred_xgb = dtc.predict(input_val)
RMSE_xgb = RMSE(y_pred_xgb, np.squeeze(y_val))
print('RMSE_xgb: %.10f' % RMSE_xgb)

#Q4 RNN
#Hyper Parameters
epochs = 100
batchsize = 100
timestamp = 8
inputsize = 49
learning_rate = 0.001           # learning rate

#numpy to tensor
x_tensor = torch.from_numpy(x)
x_tensor = x_tensor.float()
y_tensor = torch.from_numpy(y)
y_tensor = y_tensor.float()

x_val_tensor = torch.from_numpy(x_val)
x_val_tensor = x_val_tensor.float()
y_val_tensor = torch.from_numpy(y_val)
y_val_tensor = y_val_tensor.float()

x_test_tensor = torch.from_numpy(x_test)
x_test_tensor = x_test_tensor.float()

#Load data
train_X_dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
train_X_loader = torch.utils.data.DataLoader(train_X_dataset, batch_size=batchsize, shuffle=True)

val_X_dataset = torch.utils.data.TensorDataset(x_val_tensor, y_val_tensor)
val_X_loader = torch.utils.data.DataLoader(val_X_dataset, batch_size=1, shuffle=False)

test_X_loader = torch.utils.data.DataLoader(x_test_tensor, batch_size=1, shuffle=False)

#Creat model
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=inputsize,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(128,1)

    def forward(self, x):
        hidden_state = None
        output, hidden_state = self.rnn(x, hidden_state)
        out = self.out(output[:,-1,:])
        return out

net_rnn = RNN()
print(net_rnn)
net_rnn.to(device)

#training
optimizer = torch.optim.Adam(net_rnn.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

for epoch in range(epochs):
    loss_average = 0
    for i, data in enumerate(train_X_loader, 0):
        optimizer.zero_grad()
        inputs, labels = data[0].to(device), data[1].to(device)
        output = net_rnn(inputs)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        loss_average = loss_average + loss
        if i % 200 == 199:
            print('RNN: [%d, %5d] loss: %.5f' %
                  (epoch + 1, i + 1, loss_average/200))
            loss_average = 0

#Test on validation set
y_pred_rnn_tensor_total = []
with torch.no_grad():
    for data in val_X_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        y_pred_rnn_tensor = torch.relu(net_rnn(inputs))
        y_pred_rnn_tensor_total.append(y_pred_rnn_tensor.cpu().numpy())

y_pred_rnn = np.array(y_pred_rnn_tensor_total)
RMSE_rnn = RMSE(np.squeeze(y_pred_rnn), np.squeeze(y_val))
print('RMSE_rnn: %.10f' % RMSE_rnn)

#Q5 LSTM
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=inputsize,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        hidden_state = None
        output, hidden_state = self.lstm(x, hidden_state)
        out = self.out(output[:, -1, :])
        return out

net_lstm = LSTM()
print(net_lstm)
net_lstm.to(device)

#training
optimizer = torch.optim.Adam(net_lstm.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

for epoch in range(epochs):
    loss_average = 0
    for i, data in enumerate(train_X_loader, 0):
        optimizer.zero_grad()
        inputs, labels = data[0].to(device), data[1].to(device)
        output = net_lstm(inputs)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        loss_average = loss_average + loss
        if i % 200 == 199:
            print('LSTM: [%d, %5d] loss: %.5f' %
                  (epoch + 1, i + 1, loss_average/200))
            loss_average = 0

#Test on validation set
y_pred_lstm_tensor_total = []
with torch.no_grad():
    for data in val_X_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        y_pred_lstm_tensor = torch.relu(net_lstm(inputs))
        y_pred_lstm_tensor_total.append(y_pred_lstm_tensor.cpu().numpy())

y_pred_lstm = np.array(y_pred_lstm_tensor_total)
RMSE_lstm = RMSE(np.squeeze(y_pred_lstm), np.squeeze(y_val))
print('RMSE_lstm: %.10f' % RMSE_lstm)

#Form a table
Table = PrettyTable()
Table.field_names = ['RMSE_Q1','RMSE_reg','RMSE_xgb','RMSE_rnn','RMSE_lstm']
Table.add_row([RMSE_Q1, RMSE_reg, RMSE_xgb, RMSE_rnn, RMSE_lstm])
print(Table)

#Q6 Predict Test set
y_pred_test_tensor_total = []
with torch.no_grad():
    for data in test_X_loader:
        inputs = data.to(device)
        y_pred_test_tensor = torch.relu(net_rnn(inputs))
        y_pred_test_tensor_total.append(y_pred_test_tensor.cpu().numpy())

y_pred_test = np.array(y_pred_test_tensor_total)
y_pred_test = np.squeeze(y_pred_test)

#Write my prediction into CSV file
n_test = len(x_test)
with open("labels.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(n_test):
        writer.writerow([y_pred_test[i]])
print()
