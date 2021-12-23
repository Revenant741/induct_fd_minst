import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self,args):
        super(Net, self).__init__()
        self.device = args.device
        self.batch_size = args.batch
        self.size_middle = args.size_middle
        self.seq_len = 28              # 画像の Height を時系列のSequenceとしてRNNに入力する
        self.feature_size = 28         # 画像の Width を特徴量の次元としてRNNに入力する
        self.hidden_layer_size = args.size_middle   # 隠れ層のサイズ
        self.rnn_layers = 1            # RNNのレイヤー数　(RNNを何層重ねるか)
        self.x_1 = torch.zeros(self.batch_size, self.size_middle).to(self.device) 
        self.x_2 = torch.zeros(self.batch_size, self.size_middle).to(self.device) 
        self.simple_rnn = nn.RNN(input_size = self.feature_size,
                                 hidden_size = self.hidden_layer_size,
                                 num_layers = self.rnn_layers) 
        self.fc = nn.Linear(self.hidden_layer_size, 10)
        
    def initHidden(self): # RNNの隠れ層 hidden を初期化
        hedden = torch.zeros(self.rnn_layers, self.batch_size, self.hidden_layer_size).to(self.device) 
        return hedden

    def forward(self,x,binde1,binde2,binde3,binde4):
        self.hidden = self.initHidden()
        
        x = x.view(self.batch_size, self.seq_len, self.feature_size)  # (Batch, Cannel, Height, Width) -> (Batch, Height, Width) = (Batch, Seqence, Feature)
                                                                 # 画像の Height を時系列のSequenceに、Width を特徴量の次元としてRNNに入力する
        x = x.permute(1, 0, 2)                                   # (Batch, Seqence, Feature) -> (Seqence , Batch, Feature)
        #print(x.shape)
        rnn_out, h_n = self.simple_rnn(x, self.hidden)           # RNNの入力データのShapeは(Seqence, Batch, Feature)
                                                                 # (h_n) のShapeは (num_layers, batch, hidden_size)
        x = h_n[-1,:,:]                                          # RNN_layersの最後のレイヤーを取り出す (l, B, h)-> (B, h)
        x = self.fc(x)
        return x,self.x_1,self.x_2