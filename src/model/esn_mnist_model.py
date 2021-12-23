import torch
import torch.nn as nn
import math
import numpy as np
import random

#内部状態を観測出来るようにした拘束条件付きESN
class Binde_ESN_mnist_Model(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.size_middle = args.size_middle
    self.batch_size = args.batch
    self.device = args.device
    self.seq_len = 28              # 画像の Height を時系列のSequenceとしてRNNに入力する
    self.feature_size = 28         # 画像の Width を特徴量の次元としてRNNに入力する
    self.binde_esn = Reservior(args)     
    self.fc = nn.Linear(self.size_middle, 10)

  def forward(self,x,binde1,binde2,binde3,binde4):
    batch_size = x.shape[0]
    # (Batch, Cannel, Height, Width) -> (Batch, Height, Width) = (Batch, Seqence, Feature)
    x = x.view(batch_size, self.seq_len, self.feature_size)
    # 画像の Height を時系列のSequenceに、Width を特徴量の次元としてESNに入力する
    x = x.permute(1, 0, 2)   
    #ESNの入力データのShapeは(Seqence, Batch, Feature) = (Height, Batch, Width)
    #print(f'input_size={x.shape}')
    inputsout = torch.zeros(self.batch_size,128).to(self.device)
    outputsout = torch.zeros(self.batch_size,128).to(self.device) 
    for i in range(self.seq_len):
      x_1, x_2 = self.binde_esn(x[i,:,:],binde1,binde2,binde3,binde4)
      #inputsout=torch.cat([inputsout,x_1],0)
      #outputsout=torch.cat((outputsout,x_2),0)
    output = self.fc(x_2)
    #時系列出力におけるInputNeuronsとOutputNeuronsの出力記録
    inputsout = x_1
    outputsout = x_2
    return output, inputsout, outputsout

  def initHidden(self):#隠れ層の初期化
    self.binde_esn.x_1 = torch.zeros(self.batch_size, self.size_middle).to(self.device) 
    self.binde_esn.x_2 = torch.zeros(self.batch_size, self.size_middle).to(self.device) 

class Reservior(nn.Module):
  def __init__(self,args):
    super().__init__()
    self.size_in = args.size_in
    self.size_middle = args.size_middle
    self.size_out = args.size_out
    self.batch_size = args.batch
    self.device = args.device
    self.x_1 = torch.zeros(self.batch_size, self.size_middle).to(self.device) 
    self.x_2 = torch.zeros(self.batch_size, self.size_middle).to(self.device) 
    self.w_in = nn.Parameter(torch.Tensor(self.size_in,self.size_middle))
    self.w_res1 = nn.Parameter(torch.Tensor(self.size_middle,self.size_middle))
    self.w_res12 = nn.Parameter(torch.Tensor(self.size_middle,self.size_middle))
    self.w_res2 = nn.Parameter(torch.Tensor(self.size_middle,self.size_middle))
    self.w_res21 = nn.Parameter(torch.Tensor(self.size_middle,self.size_middle))
    self.b_in = nn.Parameter(torch.Tensor(self.size_middle))
    self.b_x1 = nn.Parameter(torch.Tensor(self.size_middle))
    self.b_res12 = nn.Parameter(torch.Tensor(self.size_middle))
    self.b_x2 = nn.Parameter(torch.Tensor(self.size_middle))
    self.b_res21 = nn.Parameter(torch.Tensor(self.size_middle))
    self.reset_parameters(self.w_in,self.b_in)
    self.reset_parameters(self.w_res1,self.b_x1)
    self.reset_parameters(self.w_res12,self.b_res12)
    self.reset_parameters(self.w_res2,self.b_x2)
    self.reset_parameters(self.w_res21,self.b_res21)

  def forward(self, x, binde1, binde2, binde3, binde4):
    self.x_1 = torch.matmul(x,self.w_in)+self.b_in+torch.matmul(self.x_1,torch.mul(self.w_res1,binde1))+self.b_x1+torch.matmul(self.x_2,torch.mul(self.w_res21,binde2))+self.b_res21
    self.x_2 = torch.matmul(self.x_1,torch.mul(self.w_res12,binde3))+self.b_res12+torch.matmul(self.x_2,torch.mul(self.w_res2,binde4))+self.b_x2
    self.x_1 = torch.tanh(self.x_1)
    self.x_2 = torch.tanh(self.x_2)
    return self.x_1, self.x_2

  def reset_parameters(self, weight,bias):
    #重みの初期値
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    #バイアスの初期値
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in)
    nn.init.uniform_(bias, -bound, bound)


