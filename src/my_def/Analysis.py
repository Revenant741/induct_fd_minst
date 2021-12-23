import argparse
import torch
from input import inputdata
import pickle
import matplotlib.pyplot as plt
import os
import csv
import cloudpickle

class Analysis:
  def __init__(self,args):
    self.args = args
    self.name = args.write_name

  def make_image(self, epoch, sp_accuracy, sp_loss, tp_accuracy, tp_loss):
    name = self.name
    point = 'src/img/'
    #精度のグラフ
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epoch, sp_accuracy, label="spatial")
    plt.plot(epoch, tp_accuracy, label="tempral")
    #plt.ylim(0,0.7)
    plt.legend(loc=0)
    plt.savefig(f''+point+name+'_acc.png')
    #誤差のグラフ
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.ylim(0,1.5)
    plt.plot(epoch, sp_loss, label="spatial")
    plt.plot(epoch, tp_loss, label="tempral")
    plt.legend(loc=0)
    plt.savefig(f''+point+name+'_loss.png')

  def save_to_data(self, model, sp_accuracy, sp_loss, tp_accuracy, tp_loss):
    name = self.name
    point = 'src/data/'
    #torch.save(model.to('cpu').state_dict(), point+name+'_model.pth')
    with open(f''+point+name+'model.pkl', 'wb') as f:
      cloudpickle.dump(model, f)
    with open(f''+point+name+'_sp_acc.csv', 'w') as f:
        writer = csv.writer(f)
        for accuracy1 in sp_accuracy:
            writer.writerow([accuracy1])
    with open(f''+point+name+'_tp_acc.csv', 'w') as f:
        writer = csv.writer(f)
        for accuracy2 in tp_accuracy:
            writer.writerow([accuracy2])
    with open(f''+point+name+'_sp_loss.csv', 'w') as f:
        writer = csv.writer(f)
        for loss1 in sp_loss:
            writer.writerow([loss1])
    with open(f''+point+name+'_tp_loss.csv', 'w') as f:
        writer = csv.writer(f)
        for loss2 in tp_loss:
            writer.writerow([loss2])

  #遺伝的アルゴリズムにおけるデータの保存
  def ga_save_to_data(self,Models,Acc,Loss,G,W):
    point = 'src/data/'
    #torch.save(model.to('cpu').state_dict(), point+name+'model.pth')
    #重みを一気に保存
    with open(f''+point+self.name+'_model.pkl', 'wb') as f:
      cloudpickle.dump(Models, f)
    #結合
    with open(f''+point+self.name+'_binde.dat', 'wb') as f:
        pickle.dump(W, f)
    #精度
    with open(f''+point+self.name+'_acc.csv', 'w') as f:
        writer = csv.writer(f)
        for acc in Acc:
          writer.writerow([acc])
    #誤差
    with open(f''+point+self.name+'_loss.csv', 'w') as f:
          writer = csv.writer(f)
          for losses in Loss:
            writer.writerow([losses])
    #世代
    with open(f''+point+self.name+'_generation.csv', 'w') as f:
        writer = csv.writer(f)
        for self.args.generation in G:
          writer.writerow([self.args.generation])
  
  def save_to_mutual(self,IN,OUT):
    name = self.name
    point = 'src/data/'
    with open(f''+point+name+'_in.csv', 'w') as f:
        writer = csv.writer(f)
        for input_neurons in IN:
            writer.writerow([input_neurons])
    with open(f''+point+name+'_out.csv', 'w') as f:
        writer = csv.writer(f)
        for out in OUT:
            writer.writerow([out])
      
  def save_to_var(self,var):
    name = self.name
    point = 'src/data/'
    with open(f''+point+name+'_var.csv', 'w') as f:
        writer = csv.writer(f)
        for var1 in var:
            writer.writerow([var1])

  def mutual_plot(self,in_x,in_y,out_x,out_y):
    name = self.name
    point = 'src/img/'
    plt.figure()
    #plt.scatter(in_x,in_y, c='blue',label="input neurons")
    #plt.scatter(out_x,out_y, c='red',label="output neurons")
    plt.scatter(in_x,in_y,c='blue')
    plt.scatter(out_x,out_y,c='red')
    #plt.xlabel('spatial information',fontsize=15)
    #plt.ylabel('temporal information',fontsize=15)
    #plt.legend(loc='upper right')
    plt.xlim(0,0.7)
    plt.ylim(0,0.7)
    plt.savefig(f''+point+name+'_mutial_info.png')
    plt.savefig(f''+point+name+'_mutial_info.svg')
    plt.savefig(f''+point+name+'_mutial_info.pdf')

  def __del__(self):
    pass