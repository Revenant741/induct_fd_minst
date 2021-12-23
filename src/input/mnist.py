import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def setup_mnist(args):
  #データ・セットの準備
  #データ前処理 transform を設定
  transform = transforms.Compose(
      [transforms.ToTensor(),# Tensor変換とshape変換 [H, W, C] -> [C, H, W]
      transforms.Normalize((0.5, ), (0.5, ))]) # 標準化 平均:0.5  標準偏差:0.5
  
  #訓練用(train + validation)のデータセット サイズ:(channel, height, width) = (1,28,28) 60000枚
  trainset = torchvision.datasets.MNIST(root='./data', 
                                          train=True,
                                          download=False,
                                          transform=transform)
  
  #訓練用データセットを train と val にshuffleして分割する
  train_dataset, val_dataset = torch.utils.data.random_split(trainset, [40000, 20000],generator=torch.Generator().manual_seed(42))
    
  #テスト(test)用のデータセット サイズ:(channel, height, width) = (1,28,28) 10000枚
  testset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          download=False, 
                                          transform=transform)
  
  #データローダーの作成
  
  #訓練用データセットのデータローダーの作成
  trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=args.batch,
                                              shuffle=True,
                                              num_workers=2)
  
  #評価用データセットのデータローダーの作成
  valloader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=args.batch,
                                              shuffle=False,
                                              num_workers=2)

  #テスト用データセットのデータローダーの作成
  testloader = torch.utils.data.DataLoader(testset, 
                                              batch_size=args.batch,
                                              shuffle=False, 
                                              num_workers=2)
  #classes = tuple(np.linspace(0, 9, 10, dtype=np.uint8))
  # 辞書型変数にまとめる
  dataloaders_dict = {"train": trainloader, "val": valloader, "test": testloader}
  
  print("train_dataset size = {}".format(len(train_dataset)))
  print("val_dataset size = {}".format(len(val_dataset)))
  print("test_dataset size = {}".format(len(testset)))
  import_mnist_test(args,dataloaders_dict)
  return trainloader, valloader, testloader,dataloaders_dict

def import_mnist_test(args,dataloaders_dict):
  batch_iterator = iter(dataloaders_dict["train"])  # イテレータに変換
  imges, labels = next(batch_iterator)  # 1番目の要素を取り出す
  print("imges size = ", imges.size())
  print("labels size = ", labels.size())

  #試しに1枚 plot してみる
  plt.imshow(imges[0].numpy().reshape(28,28), cmap='gray')
  plt.title("label = {}".format(labels[0].numpy()))
  plt.show()
  plt.savefig('src/img/'+args.write_name+'_mnist_test_plot.png')