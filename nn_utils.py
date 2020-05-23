import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, embed_size=300, hidden_size=1):
        super(Attention, self).__init__()
        self.model = nn.LSTM(embed_size, hidden_size,
                             bidirectional=True, batch_first=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, source_lengths):
        hidden_states, _ = self.model(x)
        hidden_states = torch.tensor([(hidden_states[i][:source_lengths[i]][:]).mean(dim=2)
                                      for i in range(hidden_states.size(0))])
        attention = self.softmax(hidden_states)
        diag = torch.stack([torch.diag(attention[i])
                            for i in attention.shape[0]], dim=0)
        x = torch.bmm(diag, x)
        return x


class FeatureExtaction(nn.Module):
    def __init__(self, embed_size=300, hidden_size=150):
        super(FeatureExtaction, self).__init__()
        self.model = nn.LSTM(embed_size, hidden_size,
                             bidirectional=True, batch_first=True)
        self.tanh = nn.Tanh()

    def forward(self, x, source_lengths):
        hidden_states, _ = self.model(x)
        h_k = torch.Tensor([hidden_states[i][source_lengths[i]-1][:]
                            for i in range(hidden_states.size(0))])
        hidden_states = torch.tensor([(hidden_states[i][:source_lengths[i]][:]).mean(dim=2)
                                      for i in range(hidden_states.size(0))])
        features = self.tanh(torch.cat((hidden_states, h_k), dim=1))
        return features


class EmotionRegression(nn.Module):
    def __init__(self, features_size=600):
        super(EmotionRegression, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(features_size, 1),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, features_size=300):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(features_size, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)
