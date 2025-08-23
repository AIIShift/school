import datetime, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import DIAGNOSIS, EarlyStopping, RMSE, MSE, MAE, MAPE
from dataloader import DataSet
from model import DKNN

datafile ='SRF5s1_111_rand0.1s10_s5.csv'  #  sampled dataset in folder "Data/dataset".
# The example "SRF5s1_112_rand0.05s10_s5.csv" denotes sampling at a rate of 5%, with a random seed of 10, from the random field "SRF5s1_112.csv", further partitioned into training and testing sets using a random seed of 5.

batch_size = 128  # batch size
lr = 0.0001  # learning rate
hidden_neurons = [4, 256, 16]  # [input dimension, model dimension, trend dimension]. Note that the input dimension should be equal to the number of all variables (auxiliary and target) in the dataset
pe_weight = 0.8  # weight of positional vector
top_k = 400  # top k nearest neighbors
loss_type = 'rmse'  # loss function type, default: rmse, options: rmse, mse, mae, mape
optim_type='adam'  # optimizer type, default: adam, options: adam, sgd
if_summary = True  # if save the training summary or not

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print(device)

datapath = 'Data/dataset/' + datafile
# read the data
data = pd.read_csv(datapath)
print(data.head())

# Load data via class DataSet
dataset = DataSet(data)

# Scaling the data
data_scaler = dataset.scaler_data()

# get the scaled train data and  test data
data_train_scaler = data_scaler['train']
data_test_scaler = data_scaler['test']
print('data_train_scaler:')
print(data_train_scaler.describe())

# get the dataloader
train_dataloader = DataLoader(data_train_scaler.values.astype(float), shuffle=True, batch_size=batch_size, drop_last=False)
test_dataloader = DataLoader(data_test_scaler.values.astype(float), shuffle=False, batch_size=batch_size, drop_last=False)

# Take the training set as known point data (observed locations)
known_coods_scaler = data_train_scaler.values[:, 1:3]
known_feature_scaler = data_train_scaler.values[:, 3:]

# tramsform the data to tensor
known_coods_scaler = torch.from_numpy((known_coods_scaler.astype(float))).to(torch.float32).to(device)
known_feature_scaler = torch.from_numpy(known_feature_scaler.astype(float)).to(torch.float32).to(device)

# Define the DKNN model
modelname = 'DKNN'
d_input, d_model, d_trend = hidden_neurons
model = DKNN(d_input=d_input, d_model=d_model, known_num=dataset.train_num, d_trend=d_trend, top_k=top_k, pe_weight=pe_weight)

# Calculate positional embeddings before training to increase the speed of training
model.cal_pe_know(known_feature_scaler, known_coods_scaler)
model.cal_pe_unknow(torch.from_numpy(data_test_scaler.values[:, 3:3+d_input].astype(float)).to(torch.float32).to(device),
                    torch.from_numpy(data_test_scaler.values[:, 1:3].astype(float)).to(torch.float32).to(device))
model.to(device)

##### loss function #####
if loss_type == 'mae':
    criterion = MAE()
elif loss_type == 'mse':
    criterion = MSE()
elif loss_type == 'rmse':
    criterion = RMSE()
elif loss_type == 'mape':
    criterion = MAPE()

##### optimizer #####
if optim_type == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
elif optim_type == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0)

# learning rate decay
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, verbose=True, patience=5, min_lr=1e-6)

# get the datafile name
datafile = datapath[datapath.rfind('/')+1:]
datafilename = datafile[0:datafile.rfind('.')]

# get the save path
start_time = datetime.datetime.now()
train_info = start_time.strftime("%m%d_%H%M%S")
save_dir = './results/' +  modelname + '/' + datafilename + '/' + train_info

# get the summary path and build the summarywriter
if if_summary:
    summarypath = save_dir + '/summary/'
    writer = SummaryWriter(summarypath)
    print('summary path: \n' + summarypath)



# get the name of radom field file
RFname = datafilename[0: datafilename[0: datafilename.rfind('_')].rfind('_')]
# read the random field
#RFdata = pd.read_csv('Data/random_field/' + RFname + '.csv')
RFdata = pd.read_csv(r'D:\PyCharm\pythonProject\HN-interpolation\Data\random_field\SRF5s1_112shuj0.05.csv')
print(RFdata.head())

RFdata_scaler = RFdata.copy()

# Remove the redundant column
if 'trend' in RFdata_scaler.columns:
    RFdata_scaler = RFdata_scaler.drop('trend',axis=1, inplace=False)

# scale the data
RFdata_scaler.iloc[:, 1:3] = dataset.scaler_coods.transform(RFdata_scaler.values[:, 1:3])
RFdata_scaler.iloc[:, 3:-1] = dataset.scaler_features.transform(RFdata_scaler.values[:, 3:-1])
RFdata_scaler.iloc[:,-1] = dataset.scaler_label.transform(RFdata_scaler.values[:, -1].reshape(-1,1))

# get the dataloader
RF_dataloader = DataLoader(RFdata_scaler.values, shuffle=False, batch_size=500, drop_last=False)
print(RFdata_scaler.describe())

# Model initialization
net = DKNN(d_input=d_input, d_model=d_model, known_num=dataset.train_num, d_trend=d_trend, top_k=top_k, pe_weight=pe_weight)

# load the best model parameters
net.load_state_dict(torch.load(r'D:\PyCharm\pythonProject\HN-interpolation\results\DKNN\SRF5s1_111_rand0.1s10_s5\0627_193752\checkpoint.pth', map_location=torch.device(device)))
print("111")
# Calculate positional embedding before training to increase the speed of training
net.cal_pe_know(known_feature_scaler, known_coods_scaler)
print("222")
net.cal_pe_unknow(torch.from_numpy(RFdata_scaler.values[:, 3:3+d_input].astype(float)).to(torch.float32).to(device),
                    torch.from_numpy(RFdata_scaler.values[:, 1:3].astype(float)).to(torch.float32).to(device))
print("333")
net.to(device)
print("444")
RFoutput = []
with torch.no_grad():
    net.eval()  # set the model to evaluation mode
    for i in tqdm(RF_dataloader):
        # model input
        i = i.to(torch.float32)
        input_feature = i[:, 3:3+d_input].to(device)
        input_feature[:,-1] = 0
        input_coods = i[:, 1:3].to(device)
        if net.pe_unknow is not None:
            input_pe = net.pe_unknow[i[:,0].type(torch.long)]  # The positional embeddings have been calculated in advance, get it according to the index
        else:
            input_pe = net.position_rep(input_feature, input_coods[:, 0], input_coods[:, 1])  # The positional embeddings are not calculated in advance
        # model execution
        output, _ = net(input_coods, input_feature, input_pe, known_coods_scaler, known_feature_scaler)
        RFoutput.extend(output.cpu().detach().numpy())

# reverse the output
RFoutput_inverse = dataset.scaler_label.inverse_transform(np.array(RFoutput).reshape(-1,1))

# diagnose the reversed output
RF_diag_inverse = DIAGNOSIS(RFoutput_inverse, RFdata['target'].values.reshape(-1,1))
RF_rmse_inverse, RF_mse_inverse, RF_mae_inverse, RF_mape_inverse = RF_diag_inverse.get()

# print diagnostic results
print('MAE/RMSE/MAPE: {:.2f}/{:.2f}/{:.2f}%, RF_mse:{:.4f}, RF_R2:{:.4f}'.format(RF_mae_inverse, RF_rmse_inverse, RF_mape_inverse*100, RF_mse_inverse, RF_diag_inverse.v_r2))

##### save the predict result and diagnostic results #####
# save the predict result
RFdata['predict'] = np.array(RFoutput_inverse)
RFdata.to_csv(save_dir + '/RFresult.csv', index=False)

# save diagnostic results
with open(save_dir + '/RFresult_diag.txt', 'w') as f:
    f.write('-----------Diagnosis-------------')
    f.write('\rinverse: MAE/RMSE/MAPE: {:.2f}/{:.2f}/{:.2f}%, RF_mse:{:.4f}, RF_R2:{:.4f}'.format(RF_mae_inverse, RF_rmse_inverse, RF_mape_inverse*100, RF_mse_inverse, RF_diag_inverse.v_r2))

print(RFdata.head())

def visualize_field_pre(filepath):
    ##### Visualize the true and predicted random field #####
    data = pd.read_csv(filepath)
    columns=['target', 'predict']
    titles=['true', 'prediction']
    fig = plt.figure(figsize=(15,5))
    for i in range(2):
        ax = plt.subplot(1, 2, i+1)
        plt.imshow(data[columns[i]].values.reshape(100, 100))
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.title(titles[i], fontsize=16)
    plt.tight_layout()
    plt.savefig(filepath[0: filepath.rfind('/')] + '/RFresult_visualize.png', dpi=300, bbox_inches='tight')

visualize_field_pre(save_dir + '/RFresult.csv')