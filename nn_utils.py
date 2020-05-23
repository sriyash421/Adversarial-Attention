import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, embed_size=300, hidden_size=1):
        super(Attention, self).__init__()
        self.model = nn.LSTM(embed_size, hidden_size,
                             bidirectional=False, batch_first=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, source_lengths):
        hidden_states, _ = self.model(x)
        enc_masks = torch.zeros_like(hidden_states, dtype=torch.uint8)
        for i in range(len(source_lengths)) :
            enc_masks[i][:source_lengths[i]][0] = 1
        hidden_states.data.masked_fill_(enc_masks, -float('inf'))
        attention = self.softmax(hidden_states.squeeze())
        diag = torch.stack([torch.diag(attention[i])
                            for i in range(attention.shape[0])], dim=0)
        x = torch.bmm(diag, x)
        return x

# input_ = torch.rand((64, 32, 300))
# print("input: {}".format(input_.shape))
# source_lengths = [32] * 64
# attn = Attention()(input_, source_lengths)
# print("Weighted input: {}".format(attn.shape))


class FeatureExtaction(nn.Module):
    def __init__(self, embed_size=300, hidden_size=150):
        super(FeatureExtaction, self).__init__()
        self.model = nn.LSTM(embed_size, hidden_size,
                             bidirectional=True, batch_first=True)
        self.tanh = nn.Tanh()

    def forward(self, x, source_lengths):
        hidden_states, _ = self.model(x)
        h_k = torch.stack([hidden_states[i][source_lengths[i]-1][:]
                            for i in range(hidden_states.shape[0])], dim=0)
        hidden_states = torch.stack([(hidden_states[i][:source_lengths[i]][:]).mean(dim=0)
                                      for i in range(hidden_states.size(0))], dim=0)
        features = self.tanh(torch.cat((hidden_states, h_k), dim=1))
        return features

# feat = FeatureExtaction()(attn, source_lengths)
# print("Features: {}".format(feat.shape))

class EmotionRegression(nn.Module):
    def __init__(self, features_size=1200):
        super(EmotionRegression, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(features_size, 1),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.model(x)

# reg = EmotionRegression()(torch.cat((feat,feat), dim=1))
# print("Reg: {}".format(reg.shape))

class Discriminator(nn.Module):
    def __init__(self, features_size=600):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(features_size, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# p = Discriminator()(feat)
# print("p: {}".format(p.shape))