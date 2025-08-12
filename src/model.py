import torch, torch.nn as nn
class PilotNetSNN(nn.Module):
    def __init__(self, in_channels=3, conv_channels=(24,36,48), fc_dims=(100,50,10)):
        super().__init__()
        c1,c2,c3 = conv_channels
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, c1, 5, stride=2), nn.ReLU(),
            nn.Conv2d(c1, c2, 5, stride=2), nn.ReLU(),
            nn.Conv2d(c2, c3, 5, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d((6,8)), nn.Flatten()
        )
        flat = c3*6*8
        self.head = nn.Sequential(nn.Linear(flat, fc_dims[0]), nn.ReLU(),
                                  nn.Linear(fc_dims[0], fc_dims[1]), nn.ReLU(),
                                  nn.Linear(fc_dims[1], 1))
    def forward(self, x):
        if x.dim()==4:
            feats = self.backbone(x)
        elif x.dim()==5:
            N,T,C,H,W = x.shape
            feats = self.backbone(x.view(N*T,C,H,W)).view(N,T,-1).mean(1)
        else:
            raise ValueError(f"Forma no soportada: {tuple(x.shape)}")
        return self.head(feats).squeeze(1)
