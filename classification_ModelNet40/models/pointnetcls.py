
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import cal_loss

class PointNetCls(nn.Module):
    def __init__(self, d=3, num_classes=40):
        super(PointNetCls, self).__init__()
        self.d = d
        self.num_classes = num_classes
        self.encoder = nn.Sequential(
            nn.Conv1d(d, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, X):
        B, d, N = X.shape
        X = self.encoder(X)
        # print('X:', X.shape)
        X = F.max_pool1d(X, N).squeeze(2)
        X = self.decoder(X)
        return X
    
    def get_loss(self, X, y, loss_type = 'cross_entropy+orthogonality', smoothing=True):
        logits = self(X)
        if loss_type == 'cross_entropy':
            return cal_loss(logits, y, smoothing), logits
        elif loss_type == 'cross_entropy+orthogonality':
            B, d, N = X.shape
            N_reg = 1000
            bbox_inputs = torch.FloatTensor(B, d, N_reg).uniform_(-1, 1)
            bbox_inputs = bbox_inputs.to(X.device) 
            bbox_features = self.encoder(bbox_inputs) 
            bbox_matirx = torch.bmm(bbox_features, bbox_features.permute(0, 2, 1)) / N_reg # (B, 1024, 1024)
            I = torch.eye(1024).to(X.device).unsqueeze(0).expand(B, -1, -1)
            orthogonality_loss = F.mse_loss(bbox_matirx, I) * 0.001
            return cal_loss(logits, y, smoothing) + orthogonality_loss, logits
        else:
            raise ValueError('Unknown loss type')
 

if __name__ == '__main__':
    data = torch.rand(2, 3, 1024)
    print("===> testing PointNet ...")
    model = PointNetCls()
    out = model(data)
    print(out.shape)

