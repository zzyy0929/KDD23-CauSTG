import torch.optim as optim
import torch.nn.functional as F
from model import *
import util
class trainer():
    def __init__(self, scaler, in_dim, seq_length, num_nodes, nhid , dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit):
        self.model = CauSTG(device, num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, in_dim=in_dim, out_dim=seq_length, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5
        self.lambda0,self.lambda1,self.lambda2 = 1e-5,1e-5,1e-5
    def diff(self,x):
        return x[..., 1:] - x[..., :-1]
    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input,(1,0,0,0))
        output,pred_trend,pred_season = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]-》batch_size,1,num_nodes,12]
        real = torch.unsqueeze(real_val,dim=1)
        with torch.no_grad():
            real_trend, real_season = self.model.decomposed_sea_tre(real.cuda())
        loss_sea = util.masked_mape(pred_season,real_season,0.0)
        loss_tre = torch.mean(F.cosine_similarity(pred_trend,real_trend)) + torch.mean(F.cosine_similarity(self.diff(pred_trend),self.diff(real_trend)))
        loss_reg = torch.mean(F.cosine_similarity(pred_trend,pred_season))
        # print(loss_sea.shape,loss_tre.shape,loss_reg.shape)
        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real, 0.0)
        loss_total = loss + self.lambda0*loss_sea + self.lambda1*loss_tre + self.lambda2*loss_reg
        loss_total.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse

    def eval(self, input, real_val):
        self.model.eval()
        input = nn.functional.pad(input,(1,0,0,0))
        output,trend,season = self.model(input)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = util.masked_mape(predict,real,0.0).item()
        rmse = util.masked_rmse(predict,real,0.0).item()
        return loss.item(),mape,rmse
