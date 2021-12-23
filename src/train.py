import argparse
from functools import total_ordering
from input import mnist
from model import esn_mnist_model as Model
from model import rnn as rnn
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
import sklearn.metrics
import time

def add_arguments(parser):
  parser.add_argument('--device', type=str, default="cuda:0", help='cpu or cuda')
  parser.add_argument('--batch', type=int,default=80, help='batch_size')
  parser.add_argument('--epoch', type=int, default=10)
  parser.add_argument('--size_in', type=int,default=28, help='middle_layer_size')
  parser.add_argument('--size_middle', type=int,default=128, help='middle_layer_size')
  parser.add_argument('--size_out', type=int,default=10, help='output_layer_size')
  parser.add_argument('--write_name', default='mnist_train', help='savename')

def random_binde(args):
  binde1 = torch.randint(0, 2, (args.size_middle, args.size_middle)).to(args.device)  
  binde2 = torch.randint(0, 2, (args.size_middle, args.size_middle)).to(args.device)  
  binde3 = torch.randint(0, 2, (args.size_middle, args.size_middle)).to(args.device)  
  binde4 = torch.randint(0, 2, (args.size_middle, args.size_middle)).to(args.device)  
  return binde1, binde2, binde3, binde4
  
def all_conect_binde(args):
  binde1 = torch.randint(1, 2, (args.size_middle, args.size_middle)).to(args.device)  
  binde2 = torch.randint(1, 2, (args.size_middle, args.size_middle)).to(args.device)  
  binde3 = torch.randint(1, 2, (args.size_middle, args.size_middle)).to(args.device)  
  binde4 = torch.randint(1, 2, (args.size_middle, args.size_middle)).to(args.device)  
  return binde1, binde2, binde3, binde4

class Adam_mnist_train:
  def __init__(self,args,model,optimizer):
    self.model = model
    self.args = args    
    self.optimizer = optimizer

  def train(self,trainloader,valloader,binde1, binde2, binde3, binde4):
    model = self.model
    model.to(self.args.device)
    #print(model)
    loss_func = nn.CrossEntropyLoss()
    optimizer = self.optimizer
    #epoch数のカウント
    for epoch in range(self.args.epoch):
      #trainloaderからステップ数と入力データ，正解データをインポート
      #学習
      train_total_accuracy = 0
      train_total_loss = 0
      model = model.train()
      for step,(input,train_ans) in enumerate(trainloader):
        #optimizerの初期化
        optimizer.zero_grad()
        #GPU使用の明示
        input = input.to(self.args.device)
        train_ans = train_ans.to(self.args.device)
        #順伝搬の計算
        #print(train_ans[0:10])
        output, x_1, x_2 = model(input, binde1, binde2, binde3, binde4)
        pred_x = torch.max(output, 1)[1].cpu().data.numpy()
        #正解率と誤差の算出
        train_accuracy = (accuracy_score(train_ans.cpu().data.numpy(), pred_x))
        train_loss = loss_func(output, train_ans)
        train_total_accuracy += train_accuracy
        train_total_loss += train_loss
        #print(train_accuracy)
        #誤差の計算
        loss = loss_func(output, train_ans)
        loss.backward()
        #モデルの内部状態の初期化
        model.initHidden()
        optimizer.step()
      ave_acc = train_total_accuracy/step
      ave_loss = train_total_loss/step
      self.print_acc_loss(ave_acc,ave_loss,epoch,"train")
      #学習で必ず行う動作において精度を算出する
      #検証
      with torch.no_grad():
        model = model.eval()
        total_accuracy = 0
        total_loss = 0
        #total_in_mutual= []
        #total_out_mutual= []
        for step,(val_input,val_ans) in enumerate(valloader):
          #GPU使用の明示
          val_input = val_input.to(self.args.device)
          val_ans = val_ans.to(self.args.device)
          #print(val_ans[0:10])
          #評価
          val_output, x_1, x_2 = model(val_input,binde1, binde2, binde3, binde4)
          pred_y = torch.max(val_output, 1)[1].cpu().data.numpy()
          accuracy = (accuracy_score(val_ans.cpu().data.numpy(), pred_y))
          loss = loss_func(val_output, val_ans)
          total_accuracy += accuracy
          #print(accuracy)
          total_loss += loss
          #モデルの内部状態の初期化
          model.initHidden()
          #相互情報量の算出
        in_mutual, out_mutual = self.mutual_info(x_1,x_2,train_ans)
        ave_acc = total_accuracy/step
        ave_loss = total_loss/step
        self.print_acc_loss(ave_acc,ave_loss,epoch,'val')
    return model, ave_acc, ave_loss,in_mutual,out_mutual

  def print_acc_loss(self,accuracy_all,loss_all,epoch,var):
    epoch_str = f'Epoch: {epoch+1}'
    train_loss_str = f'{var}_loss: {loss_all.data.cpu().numpy():.4f}'
    val_accuracy_str = f'{var}_accuracy: {accuracy_all:.2f}'
    print(f'{epoch_str} | {train_loss_str} | {val_accuracy_str}')  

  def __del__(self):
    pass

  def mutual_info(self,x1,x2,ans):    
    h_in = []
    h_out = []
    #x1=(80,120) ans=(80)
    in_neurons = x1.to('cpu').detach().numpy().copy()
    out_neurons = x2.to('cpu').detach().numpy().copy()
    bins=10
    range_x1=(-1,1)
    for j in range(self.args.size_middle):
      n_in = in_neurons[:,j]
      n_out = out_neurons[:,j]
      #n_in=(80)
      _,bins_x1  = np.histogram(n_in.flatten(), bins,range_x1)
      _,bins_x1  = np.histogram(n_out.flatten(), bins,range_x1)
      n_in_mutial = np.digitize(n_in.flatten(), bins_x1)
      n_out_mutial = np.digitize(n_out.flatten(), bins_x1)
      #n_inの中身を10段階に分類
      n_in_mutial = sklearn.metrics.mutual_info_score(n_in_mutial,ans.to('cpu').detach().numpy().copy())
      n_out_mutial = sklearn.metrics.mutual_info_score(n_out_mutial,ans.to('cpu').detach().numpy().copy())
      #相互情報量を算出
      h_in.append(n_in_mutial)
      h_out.append(n_out_mutial)
    return h_in, h_out

def save_in_mnist(input_data,input_ans):
  npimg=input_data[0].to('cpu').detach().numpy().copy()
  npimg = npimg.reshape((28,28))
  plt.figure()
  plt.imshow(npimg, cmap='gray')
  #plt.savefig("mnist_img"+{str(step)}+".png")
  plt.savefig("mnist_img.png")
  print('Label:', input_ans[0])

def main(args):
  #拘束条件の設定
  #ランダム結合
  binde1, binde2, binde3, binde4 = random_binde(args)
  #全結合
  #binde1, binde2, binde3, binde4 = all_conect_binde(args)
  #ESNの場合
  model = Model.Binde_ESN_mnist_Model(args)
  #rnnの場合（テスト用）
  #model = rnn.Net(args)
  #データセットを用意
  trainloader,valloader, testloader, dataloaders_dict = mnist.setup_mnist(args)
  optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
  Adam = Adam_mnist_train(args,model,optimizer)
  #Adam.train(trainloader,valloader,binde1, binde2, binde3, binde4)
  Adam.train(trainloader,testloader,binde1, binde2, binde3, binde4)
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  add_arguments(parser)
  args = parser.parse_args()
  print(args)
  main(args)