import argparse
import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from my_def import Analysis
import train
import copy
from input import mnist
from model import esn_mnist_model as Model
import time
#from torch.utils.tensorboard import SummaryWriter

def add_arguments(parser):
  #ESNパラメータ
  parser.add_argument('--device', type=str, default="cuda:0", help='cpu or cuda')
  parser.add_argument('--batch', type=int,default=80, help='batch_size')
  parser.add_argument('--epoch', type=int, default=10)
  parser.add_argument('--size_in', type=int,default=28, help='middle_layer_size')
  parser.add_argument('--size_middle', type=int,default=128, help='middle_layer_size')
  parser.add_argument('--size_out', type=int,default=10, help='output_layer_size')
  parser.add_argument('--write_name', default='mnist_train', help='savename')
  #GAパラメータ
  parser.add_argument('--pop', type=int, default=20, help='pop_model_number')
  parser.add_argument('--survivor', type=int, default=10, help='pop_model_number')
  parser.add_argument('--mutate_rate', default=0.25, help='mutate_rate')
  parser.add_argument('--generation', type=int, default=100, help='generation')
  parser.add_argument('--gene_length', default=128, help='pop_model_number')
  #test時
  #python3 src/ga_train.py --pop 3 --survivor 2 --write_name 'ga_test' --epoch 1
  #実行
  #python3 src/ga_train.py --pop 20 --survivor 10 --epoch 5 --device 'cuda:0' --write_name 'loss_eva_e5_p20_l10'

#世代における個体の評価
def make_one_gene(args, g, binde, trainloader, valloader, ind):
  bindes = binde
  point = args.gene_length
  point2 = args.gene_length*2
  #世代の作成
  print('========')
  print(f'=学習=')
  ga_start_time = time.time()
  #生成した接続構造分学習
  for i in range(args.pop):
    if g == 0:
      #第一世代では接続構造を生成
      binde = torch.randint(0, 2, (args.gene_length*2, args.gene_length*2)).to(args.device)
    else:
      #新しい接続構造のみ学習
      binde = torch.from_numpy(bindes[i]).clone().to(args.device)
    #モデルの実行，重み,誤差，精度を返して来る．
    print(f'--個体{i+1}--')
    #モデルの用意
    #ESNの場合
    model = Model.Binde_ESN_mnist_Model(args)
    #最適化関数用意
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    #model学習
    individual = train.Adam_mnist_train(args,model,optimizer)
    learn_start = time.time()
    #print(binde[:point,:point].shape)
    #print(binde[:point,point:point2].shape)
    #print(binde[point:point2,:point].shape)
    #print(binde[point:point2,point:point2].shape)
    model, total_accuracy, total_loss,in_mutual,out_mutual = individual.train(trainloader,valloader,binde[:point,:point],binde[:point,point:point2],binde[point:point2,:point],binde[point:point2,point:point2])
    #時間計測
    learn_finish_time = time.time() -learn_start
    print ("-----学習時間:{:.1f}".format(learn_finish_time) + "[sec]-----")
    #個体毎に分散を算出
    binde = binde.to('cpu').detach().numpy().copy()
    ind.append((total_accuracy,total_loss,binde,model,in_mutual,out_mutual))
    del individual
  #時間計測
  ga_finish_time = time.time()-ga_start_time
  print ("-----世代学習の経過時間:{:.1f}".format(ga_finish_time) + "[sec]-----")
  return ind

#選択，生き残る個体を決める関数
def evalution(ind):
  #0で精度、1で誤差，Falseで小さい順，Trueで大きい順
  #誤差
  #ind = sorted(ind, key=lambda x:x[1], reverse=False)
  #機能分化
  ind = sorted(ind, key=lambda x:x[1], reverse=True)
  return ind

#二点交叉
def tow_point_crossover(parent1, parent2,gene_length):
  child1 = copy.deepcopy(parent1)
  for i in range(4):
    r0 = random.randint(0,gene_length*2-1)
    r1 = random.randint(0,gene_length*2-1)
    r2 = random.randint(r1,gene_length*2)
    child1[r0,r1:r2]= parent2[r0,r1:r2]
  return child1

#突然変異
def mutate(parent,gene_length):
  child = copy.deepcopy(parent)
  for i in range(40):
    r1 = random.randint(0, gene_length*2-1)
    r2 = random.randint(0, gene_length*2-1)
    if child[r1][r2] == 0:
      child[r1][r2] = 1
    else:
      child[r1][r2] = 0
  return child

#交配関数
def crossbreed(args,binde,first_pop):
  #次世代の生成，生成個体の数は初代と同じ数
  while len(binde) < first_pop:
    m1 = random.randint(0,len(binde)-1)#親となる個体の決定
    m2 = random.randint(0,len(binde)-1)#親となる個体の決定
    child = tow_point_crossover(binde[m1],binde[m2],args.gene_length)#交叉処理
    #突然変異
    if random.random() < args.mutate_rate:
      m = random.randint(0,len(binde)-1)#突然変異する個体を選択
      child = mutate(binde[m],args.gene_length)
    binde.append(child)
  return binde

#保存用の処理
def for_save(args,SAVE,g,survival ,binde):
  print('========')
  print('=評価=')
  rank = 0
  #初期化(前世代の接続を不正に残さない為)
  binde = []
  Acc, Loss ,IN ,OUT, Models, G, W = SAVE
  for acc,loss,binde1,models,in_mutual,out_mutual in survival:
    #精度の可視化
    rank += 1
    print(f'-----第{rank}位--精度={acc*100:.1f}%-----')
    #優秀な個体は次世代に持ち越し
    binde.append(binde1)
    #評価用の変数
    Acc.append(acc)
    Loss.append(loss)
    Models.append(models)
    IN.append(in_mutual)
    OUT.append(out_mutual)
    G.append(g+1)
    W.append(binde1)
  analysis = Analysis.Analysis(args)
  #重み，結合，精度，誤差，世代を保存
  analysis.ga_save_to_data(Models,Acc,Loss,G,W)
  #1世代の相互情報量の記録
  analysis.save_to_mutual(IN,OUT)
  #次の世代に持ち越し
  SAVE =Acc, Loss ,IN ,OUT, Models, G, W
  return SAVE, binde

def ga_train(args,trainloader,valloader,testloader):
  #時間計測
  total_time = time.time()
  #保存用の変数用意
  Acc, Loss ,IN ,OUT, Models, G, W = [],[],[],[],[],[],[]
  SAVE =Acc, Loss ,IN ,OUT, Models, G, W
  ind = []
  binde = []
  first_pop = args.pop
  #遺伝的アルゴリズムの開始
  for g in range(args.generation):
    #世代の作成
    print(f'\n世代{g+1}')
    #個体生成，学習，評価値を保存
    ind = make_one_gene(args, g, binde, trainloader, valloader, ind)
    #評価値順にソート
    ind = evalution(ind)
    #優秀構造のみ残す
    survival = ind[0:args.survivor]
    #次世代の優秀個体として評価値を持ち越し
    ind = survival
    #優秀な構造のみ保持，保存
    SAVE, binde = for_save(args,SAVE,g,survival,binde)
    #次世代の設定
    #交配,新しい構造を生成
    binde = crossbreed(args,binde,first_pop)
    #学習する個体は新しく生成された個体のみ
    args.pop = first_pop-args.survivor
    #新しく評価が必要なもののみ追加
    binde = binde[-args.pop:]
    #合計経過時間
    total_finish = time.time()- total_time
    print ("=====TOTAL_TIME:{:.1f}".format(total_finish) + "[sec]=====")

if __name__ == '__main__': 
  #入力変数設定
  parser = argparse.ArgumentParser()
  add_arguments(parser)
  args = parser.parse_args()
  print(args)
  #拘束条件の設定
  binde1, binde2, binde3, binde4 = train.random_binde(args)
  #データセットを用意
  trainloader,valloader, testloader, dataloaders_dict = mnist.setup_mnist(args)
  #重みと構造の探索関数
  ga_train(args,trainloader,valloader,testloader)
  
  #test
  # pop = 10 #初期個体 #次世代の個体数
  # ind_learn = train.Adam_train
  # epoch = 5
  # optimizer = torch.optim.Adam
  # generation = 2
  # survivor = 5 #生き残る個体
  # name = 'ga_test'
  # ga_train(pop,ind_learn,epoch,optimizer,generation,survivor,name)
  