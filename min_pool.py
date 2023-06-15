import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt
from engine import trainer
import torch.optim as optim
from model import *
parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=1,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
#parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default='./garage/metr',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--env',type=int,default=4,help='env id')
parser.add_argument('--ifenv',type=int,default=1,help='ifenv')
args = parser.parse_args()



# def main():
#     #set seed
#     #torch.manual_seed(args.seed)
#     #np.random.seed(args.seed)
#     #load data
device = torch.device(args.device)
sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size, env=args.env,ifenv=args.ifenv)
scaler = dataloader['scaler']
supports = [torch.tensor(i).to(device) for i in adj_mx]
args.gcn_bool=True
args.addaptadj =False
args.randomadj=False
print(args)

if args.randomadj:
    adjinit = None
else:
    adjinit = supports[0]

if args.aptonly:
    supports = None

models=[]
for env in [1,2,3,4]:
    model = CauSTG(device, args.num_nodes, args.dropout, supports=supports, gcn_bool=args.gcn_bool, addaptadj=args.addaptadj, aptinit=adjinit, in_dim=args.in_dim, out_dim=args.seq_length, residual_channels=args.nhid, dilation_channels=args.nhid, skip_channels=args.nhid * 8, end_channels=args.nhid * 16)
    model.load_state_dict(torch.load(args.save+'_env'+str(env)+".pth"))
    models.append(model)
# for model in models:
#     #print(model)
#     # for param in model.parameters():
#     #     if param.requires_grad:
#     #         print(param)
#     for name, param in model.named_parameters():
#             print(f"Parameter {name}:")
#             print(param)
# 假设models是一个包含4个PyTorch模型的列表
new_model = CauSTG(device, args.num_nodes, args.dropout, supports=supports, gcn_bool=args.gcn_bool, addaptadj=args.addaptadj, aptinit=adjinit, in_dim=args.in_dim, out_dim=args.seq_length, residual_channels=args.nhid, dilation_channels=args.nhid, skip_channels=args.nhid * 8, end_channels=args.nhid * 16)
# 遍历新模型的每个参数
for name, param in new_model.named_parameters():
    # 计算四个模型的参数的均值
    mean_param = torch.mean(torch.stack([model.state_dict()[name] for model in models]), dim=0)
    min_deviation = float('inf')
    best_param = None
    for model in models:
        deviation = torch.norm(model.state_dict()[name] - mean_param)
        if deviation < min_deviation:
            min_deviation = deviation
            best_param = model.state_dict()[name]
    # 将均值复制到新模型的参数中
    param.data.copy_(best_param.clone())
torch.save(new_model.state_dict(), args.save+'_env5'+".pth")