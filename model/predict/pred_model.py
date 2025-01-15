import torch
import torch.nn as nn
import torch.nn.functional as F
import collections


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, input_dim, output_dim):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2 + input_dim, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 128),
            nn.Linear(128, output_dim)
        )

    def forward(self, input, feat_input):
        x = self.embedding(input)
        x = torch.transpose(x, 0, 1)
        _, (lstm_out, _) = self.lstm(x)
        lstm_out = torch.cat((lstm_out[0], lstm_out[1]), dim=1)
        combined = torch.cat((lstm_out, feat_input), dim=1)
        output = self.fc_layers(combined)
        output = torch.sigmoid(output)
        return output


"""
BiLSTMAttention
"""
class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim):
        super(Attention, self).__init__()
        self.step_dim = step_dim
        self.feature_dim = feature_dim
        self.w = nn.Parameter(torch.zeros(feature_dim))
        self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        eij = torch.mm(x.contiguous().view(-1, self.feature_dim), self.w.unsqueeze(1)).view(-1, self.step_dim)
        eij = eij + self.b
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        if mask is not None:
            a = a * mask
        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)
        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)

class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, input_seq_len, input_dim, output_dim):
        super(BiLSTMAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.attention = Attention(hidden_dim * 2, input_seq_len)
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2 + input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid()
        )

    def forward(self, input, feat_input):
        x = self.embedding(input)
        x, _ = self.lstm(x)
        x = self.attention(x)
        x = torch.cat((x, feat_input), 1)
        x = self.fc_layers(x)
        return x


"""
BiLSTM_4
"""
class AggregateLayer(nn.Module):
    def __init__(self, d_model=None, dropout=0.1):
        super(AggregateLayer, self).__init__()
        self.attn = nn.Sequential(collections.OrderedDict([
            ('layernorm', nn.LayerNorm(d_model)),
            ('fc', nn.Linear(d_model, 1, bias=False)),
            ('dropout', nn.Dropout(dropout)),
            ('softmax', nn.Softmax(dim=1))
        ]))

    def forward(self, context):  # contest[batch_size, aggregate_dim]
        weight = self.attn(context) # [batch_size, 1]
        weighted_context = context * weight  #[batch_size, aggregate_dim]
        return weighted_context

class GlobalPredictor(nn.Module):
    def __init__(self, d_model=None, d_h=None, d_out=None, dropout=0.5):
        super(GlobalPredictor, self).__init__()
        self.batchnorm = nn.BatchNorm1d(d_model)
        self.predict_layer = nn.Sequential(collections.OrderedDict([
            ('fc1', nn.Linear(d_model, d_h)),
            ('tanh', nn.Tanh()),
            ('dropout', nn.Dropout(dropout)),
            ('fc2', nn.Linear(d_h, d_out))
        ]))

    def forward(self, x):  # x[batch_size, aggregate_dim]
        if x.shape[0] != 1:
            x = self.batchnorm(x)
        x = self.predict_layer(x)  #[batch_size, 1]
        return x

class BiLSTMPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, input_dim, aggregate_dim, predictor_d_h):
        super(BiLSTMPredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

        # 全连接层的输出维度调整为 AggregateLayer 的输入维度
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2 + input_dim, aggregate_dim)
        )

        # 添加 AggregateLayer 和 GlobalPredictor
        self.aggregate_layer = AggregateLayer(d_model=aggregate_dim)
        self.global_predictor = GlobalPredictor(d_model=aggregate_dim, d_h=predictor_d_h, d_out=1)  # 输出维度设置为 1

    def forward(self, input, feat_input):  # input[batch_size, seq_length] ; feat_input[batch_size, num_features]
        x = self.embedding(input)  # [batch_size, seq_length, embedding_dim]
        x = torch.transpose(x, 0, 1)  # [seq_length, batch_size, embed_dim]
        _, (lstm_out, _) = self.lstm(x)  # lstm_out[2, batch_size, hidden_dim]
        lstm_out = torch.cat((lstm_out[0], lstm_out[1]), dim=1)  # [batch_size, hidden_dim*2]
        combined = torch.cat((lstm_out, feat_input), dim=1)  # [batch_size, (hidden_dim*2 + num_features)]

        combined = self.fc_layers(combined)  #[batch_size, aggregate_dim]
        aggregated = self.aggregate_layer(combined)  # [batch_size, aggregate_dim]
        output = self.global_predictor(aggregated) # [batch_size, 1]

        output = torch.sigmoid(output)  # 确保输出在 [0, 1] 区间
        return output


