import torch
import torch.nn as nn

class Attention(nn.Module) :
    def __init__(self, embed_size=300, hidden_size=1) :
        super(self, Attention).__init__()
        self.model = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x) :
        hidden_states, _ = self.model(x)
        hidden_states = hidden_states.mean(dim=-1).squeeze()
        attention = self.softmax(hidden_states)
        diag = torch.stack([torch.diag(attention[i]) for i in attention.shape[0] ], dim=0)
        x = torch.bmm(diag, x)
        return x

class FeatureExtaction(nn.Module) :
    def __init__(self, embed_size=300, hidden_size=150) :
        super(self, FeatureExtaction).__init__()
        self.model = nn.LSTM(embed_size, hidden_size, bidirectional=True, batch_first=True)
        self.tanh = nn.Tanh()
    
    def forward(self, x) :
        hidden_states, (h_k, _) = self.model(x)
        hidden_states = hidden_states.mean(dim=1).squeeze()
        features = self.tanh(torch.cat((hidden_states, h_k), dim=1))
        return features

class EmotionRegression(nn.Module) :
    def __init__(self, features_size=600):
        super(self, EmotionRegression).__init__()
        self.model = nn.Sequential(
            nn.Linear(features_size, 1),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module) :
    def __init__(self, features_size=300):
        super(self, Discriminator).__init__()
        self.model = nn.Sequential(
            nn.Linear(features_size, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.model(x)