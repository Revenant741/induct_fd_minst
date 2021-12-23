import argparse
from functools import total_ordering
from input import mnist
from model import esn_mnist_model as Model
from model import rnn as rnn
import matplotlib.pyplot as plt
import model
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
import train
from my_def import hessianfree

def add_arguments(parser):
  parser.add_argument('--device', type=str, default="cuda:0", help='cpu or cuda')
  parser.add_argument('--batch', type=int,default=1, help='batch_size')
  parser.add_argument('--epoch', type=int, default=10)
  parser.add_argument('--size_in', type=int,default=28, help='middle_layer_size')
  parser.add_argument('--size_middle', type=int,default=128, help='middle_layer_size')
  parser.add_argument('--size_out', type=int,default=10, help='output_layer_size')
  parser.add_argument('--write_name', default='mnist_train', help='savename')

def random_binde(args):
  binde1 = torch.randint(1, 2, (args.size_middle, args.size_middle)).to(args.device)  
  binde2 = torch.randint(1, 2, (args.size_middle, args.size_middle)).to(args.device)  
  binde3 = torch.randint(1, 2, (args.size_middle, args.size_middle)).to(args.device)  
  binde4 = torch.randint(1, 2, (args.size_middle, args.size_middle)).to(args.device)  
  return binde1, binde2, binde3, binde4

class HessianFree_mnist_train(train.Adam_mnist_train):
  def __init__(self,args,model,optimizer,loss_func):
    self.model = model
    self.args = args    
    self.optimizer = optimizer
    self.loss_func = loss_func

  def train(self,trainloader,valloader,binde1,binde2,binde3,binde4):
    model = self.model
    optimizer = self.optimizer(model.parameters(), use_gnm=True, verbose=True)
    model.to(args.device)
    print(model)
    loss_func = self.loss_func
    #epoch数のカウント
    for epoch in range(args.epoch):
      #trainloaderからステップ数と入力データ，正解データをインポート
      #学習
      for step,(input,train_ans) in enumerate(trainloader):
        #optimizerの初期化
        optimizer.zero_grad()
        #GPU使用の明示
        input = input.to(args.device)
        train_ans = train_ans.to(args.device)
        def closure():
          #順伝搬の計算
          output, x_1, x_2 = model(input, binde1, binde2, binde3, binde4)
          #誤差の計算
          loss = loss_func(output, train_ans)
          loss.backward(create_graph=True)
          return loss, output
        model.initHidden()
        optimizer.step(closure, M_inv=None)
      #モデルの内部状態の初期化
      #学習で必ず行う動作において精度を算出する
      with torch.no_grad():
        total_accuracy = 0
        total_loss = 0
        for step,(val_input,val_ans) in enumerate(valloader):
          #GPU使用の明示
          val_input = input.to(args.device)
          val_ans = train_ans.to(args.device)
          val_output, x_1, x_2 = model(val_input,binde1, binde2, binde3, binde4)
          pred_y = torch.max(val_output, 1)[1].cpu().data.numpy()
          #print(f'pred_y{pred_y}')
          #print(f'val_ans{val_ans}')
          #accuracy = float((pred_y == val_ans).astype(int).sum()) / float(val_ans.size)
          accuracy = (accuracy_score(val_ans.cpu().data.numpy(), pred_y))
          loss = loss_func(val_output, val_ans)
          total_accuracy += accuracy
          total_loss += loss
          #モデルの内部状態の初期化
          model.initHidden()
        accuracy_all = total_accuracy/step
        loss_all = total_loss/step
        epoch_str = f'Epoch: {epoch+1}'
        train_loss_str = f'train loss: {loss_all.data.cpu().numpy():.4f}'
        val_accuracy_str = f'val_accuracy: {accuracy_all:.2f}'
        print(f'{epoch_str} | {train_loss_str} | {val_accuracy_str}')  
        
def main(args):
  #拘束条件の設定
  binde1, binde2, binde3, binde4 = random_binde(args)
  model = Model.Binde_ESN_mnist_Model(args)
  #rnnの場合（テスト用）
  #model = rnn.Net(args)
  #データセットを用意
  trainloader,valloader, testloader, dataloaders_dict = mnist.setup_mnist(args)
  optimizer = hessianfree.HessianFree
  loss = nn.CrossEntropyLoss()
  HessianFree = HessianFree_mnist_train(args,model,optimizer,loss)
  HessianFree.train(trainloader,valloader,binde1, binde2, binde3, binde4)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  add_arguments(parser)
  args = parser.parse_args()
  print(args)
  main(args)
