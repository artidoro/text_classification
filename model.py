import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import utils
from collections import Counter
from itertools import chain

"""
Continuous Bag of Words model

Pre-trained word embeddings are summed to obtain a representation of the
sentence and a linear layer is applyied to the result

"""

class CBOW(nn.Module):
    def __init__(self, TEXT, LABEL, args):
        super(CBOW, self).__init__()
        self.vocab_size = len(TEXT.vocab)
        self.label_size = len(LABEL.vocab)
        self.embed_dim = args['embed_size']

        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        if not args['no_pretrained_vectors']:
            self.embed.weight = nn.Parameter(TEXT.vocab.vectors)
        self.fc1 = nn.Linear(self.embed_dim, self.label_size)

        self.dropout = nn.Dropout(args['dropout'])

    def forward(self, batch):
        emb_prod = self.embed(batch)
        cont_bow = torch.sum(emb_prod, 1)
        linear_separator = self.fc1(cont_bow)
        y = self.dropout(linear_separator)
        return y

"""
2 channels Convolutional Neural Netowrk
as described by Kim in the paper https://arxiv.org/pdf/1408.5882.pdf
pretrained word embeddings, 3 stride sizes for convolution layers,
ReLu activation and max polling, drop out regularization, normalization
of the linear fully connected layer
"""

class CNN(nn.Module):
    def __init__(self, TEXT, LABEL, args):
        super(CNN, self).__init__()
        self.vocab_size = len(TEXT.vocab)
        self.label_size = len(LABEL.vocab)
        self.embed_dim = args['embed_size']

        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.embed_static = nn.Embedding(self.vocab_size, self.embed_dim)
        if not args['no_pretrained_vectors']:
            self.embed.weight = nn.Parameter(TEXT.vocab.vectors)
            self.embed_static.weight = nn.Parameter(TEXT.vocab.vectors, requires_grad=False)

        self.conv1 = nn.Conv1d(300, 100, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(300, 100, 4, padding=2, stride=1)
        self.conv3 = nn.Conv1d(300, 100, 5, padding=2, stride=1)

        self.conv4 = nn.Conv1d(300, 100, 3, padding=1, stride=1)
        self.conv5 = nn.Conv1d(300, 100, 4, padding=2, stride=1)
        self.conv6 = nn.Conv1d(300, 100, 5, padding=2, stride=1)

        self.fc1 = nn.Linear(600, self.label_size)

        self.dropout = nn.Dropout(args['dropout'])

    def forward(self, batch):
        x_dynamic = self.embed(batch).transpose(1,2)
        x_static = self.embed_static(batch).transpose(1,2)

        c1 = F.relu(self.conv1(x_static))
        c2 = F.relu(self.conv2(x_static))
        c3 = F.relu(self.conv3(x_static))
        c4 = F.relu(self.conv4(x_dynamic))
        c5 = F.relu(self.conv5(x_dynamic))
        c6 = F.relu(self.conv6(x_dynamic))

        z1 = F.max_pool1d(c1, c1.size(2)).view(batch.size(0), -1)
        z2 = F.max_pool1d(c2, c2.size(2)).view(batch.size(0), -1)
        z3 = F.max_pool1d(c3, c3.size(2)).view(batch.size(0), -1)
        z4 = F.max_pool1d(c4, c4.size(2)).view(batch.size(0), -1)
        z5 = F.max_pool1d(c5, c5.size(2)).view(batch.size(0), -1)
        z6 = F.max_pool1d(c6, c6.size(2)).view(batch.size(0), -1)

        z = torch.cat((z1, z2, z3, z4, z5, z6), dim=1)

        d = self.dropout(z)
        y = self.fc1(d)
        return y

"""
Like the previous by 2 layer CNN
"""

class CNN2(nn.Module):

    def __init__(self, TEXT, LABEL, args):
        super(CNN2, self).__init__()
        self.vocab_size = len(TEXT.vocab)
        self.label_size = len(LABEL.vocab)
        self.embed_dim = args['embed_size']

        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.embed_static = nn.Embedding(self.vocab_size, self.embed_dim)
        if not args['no_pretrained_vectors']:
            self.embed.weight = nn.Parameter(TEXT.vocab.vectors)
            self.embed_static.weight = nn.Parameter(TEXT.vocab.vectors, requires_grad=False)
        else:
            self.embed_static.weight.requires_grad_(False)


        nc1 = nc2 = args['hidden_size']


        self.conv1 = nn.Conv1d(self.embed_dim, nc1, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(self.embed_dim, nc1, 4, padding=2, stride=1)
        self.conv3 = nn.Conv1d(self.embed_dim, nc1, 5, padding=2, stride=1)
        self.conv4 = nn.Conv1d(self.embed_dim, nc1, 3, padding=1, stride=1)
        self.conv5 = nn.Conv1d(self.embed_dim, nc1, 4, padding=2, stride=1)
        self.conv6 = nn.Conv1d(self.embed_dim, nc1, 5, padding=2, stride=1)

        self.conv21 = nn.Conv1d(nc1, nc2, 3, padding=1, stride=1)
        self.conv22 = nn.Conv1d(nc1, nc2, 3, padding=1, stride=1)
        self.conv23 = nn.Conv1d(nc1, nc2, 3, padding=1, stride=1)
        self.conv24 = nn.Conv1d(nc1, nc2, 3, padding=1, stride=1)
        self.conv25 = nn.Conv1d(nc1, nc2, 3, padding=1, stride=1)
        self.conv26 = nn.Conv1d(nc1, nc2, 3, padding=1, stride=1)

        self.fc1 = nn.Linear(nc2 * 6, self.label_size)

        self.dropout = nn.Dropout(args['dropout'])

    def forward(self, batch):
        x_dynamic = self.embed(batch).transpose(1,2)
        x_static = self.embed_static(batch).transpose(1,2)

        c1 = F.relu(self.conv1(x_dynamic))
        c2 = F.relu(self.conv2(x_dynamic))
        c3 = F.relu(self.conv3(x_dynamic))
        c4 = F.relu(self.conv4(x_static))
        c5 = F.relu(self.conv5(x_static))
        c6 = F.relu(self.conv6(x_static))

        z1 = F.max_pool1d(c1, 3, padding=1)
        z2 = F.max_pool1d(c2, 3, padding=1)
        z3 = F.max_pool1d(c3, 3, padding=1)
        z4 = F.max_pool1d(c4, 3, padding=1)
        z5 = F.max_pool1d(c5, 3, padding=1)
        z6 = F.max_pool1d(c6, 3, padding=1)

        c21 = F.relu(self.conv21(z1))
        c22 = F.relu(self.conv22(z2))
        c23 = F.relu(self.conv23(z3))
        c24 = F.relu(self.conv24(z4))
        c25 = F.relu(self.conv25(z5))
        c26 = F.relu(self.conv26(z6))

        z21 = F.max_pool1d(c21, c21.size(2)).view(batch.size(0), -1)
        z22 = F.max_pool1d(c22, c22.size(2)).view(batch.size(0), -1)
        z23 = F.max_pool1d(c23, c23.size(2)).view(batch.size(0), -1)
        z24 = F.max_pool1d(c24, c24.size(2)).view(batch.size(0), -1)
        z25 = F.max_pool1d(c25, c25.size(2)).view(batch.size(0), -1)
        z26 = F.max_pool1d(c26, c26.size(2)).view(batch.size(0), -1)

        z = torch.cat((z21, z22, z23, z24, z25, z26), dim=1)

        d = self.dropout(z)
        y = self.fc1(d)
        return y

"""
LSTM model

Bidirectional LSTM on top of pretrained word embeddings.
100 dimensional hidden representation output per direction, so 200 dimension
final hidden representation. Relu and Max pooling, and linear layer with
regularization.

Takes number of layers as input parameter (we tested 1 and 2 layers)
"""

class LSTM(nn.Module):
    def __init__(self, TEXT, LABEL, args):
        super(LSTM, self).__init__()
        self.vocab_size = len(TEXT.vocab)
        self.label_size = len(LABEL.vocab)
        self.embed_dim = args['embed_size']
        self.layers = args['num_layers']
        self.hidden_size = args['hidden_size']

        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        if not args['no_pretrained_vectors']:
            self.embed.weight = nn.Parameter(TEXT.vocab.vectors)
        # biderectional LSTM layer
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_size, self.layers, batch_first=True, dropout=args['dropout'], bidirectional=True)
        self.c0 = nn.Parameter(torch.zeros((self.layers * 2, 1, self.hidden_size)))
        self.h0 = nn.Parameter(torch.zeros((self.layers * 2, 1, self.hidden_size)))

        self.fc1 = nn.Linear(2 * self.hidden_size, self.label_size)

        self.dropout = nn.Dropout(args['dropout'])

    def forward(self, batch):
        h = self.h0.expand((-1, batch.size(0), -1)).contiguous()
        c = self.h0.expand((-1, batch.size(0), -1)).contiguous()

        H, _ = self.lstm(self.embed(batch), (h, c))
        a = F.relu(H)
        a = a.transpose(1,2)
        z = F.max_pool1d(a, a.size(2)).view(-1, 2*self.hidden_size)

        d = self.dropout(z)

        y = self.fc1(d)
        return y


"""
LSTM Self Attention model
"""
class LSTMSelfAttention(nn.Module):
    def __init__(self, TEXT, LABEL, args):
        super(LSTMSelfAttention, self).__init__()
        self.vocab_size = len(TEXT.vocab)
        self.label_size = len(LABEL.vocab)
        self.embed_dim = args['embed_size']
        self.layers = args['num_layers']
        self.hidden_size = args['hidden_size']

        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        if not args['no_pretrained_vectors']:
            self.embed.weight = nn.Parameter(TEXT.vocab.vectors)
        # biderectional LSTM layer
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_size, self.layers, batch_first=True, dropout=args['dropout'], bidirectional=True)
        self.c0 = nn.Parameter(torch.zeros((self.layers * 2, 1, self.hidden_size), requires_grad=True))
        self.h0 = nn.Parameter(torch.zeros((self.layers * 2, 1, self.hidden_size), requires_grad=True))

        self.fc1 = nn.Linear(2 * self.hidden_size + self.embed_dim, self.label_size)

        self.dropout = nn.Dropout(args['dropout'])

    def forward(self, batch):
        h = self.h0.expand((-1, batch.size(0), -1)).contiguous()
        c = self.h0.expand((-1, batch.size(0), -1)).contiguous()

        e = self.embed(batch)
        H, _ = self.lstm(e, (h, c))

        attn_scores = torch.bmm(H, torch.transpose(H, 1, 2))
        attn_probs = F.softmax(attn_scores, dim=2)
        attn_words = torch.bmm(attn_probs, e)

        h_pooled = F.max_pool1d(F.relu(H.transpose(1,2)), H.size(1)).squeeze()
        attn_pooled = F.max_pool1d(F.relu(attn_words.transpose(1,2)), attn_words.size(1)).squeeze()
        skip_connection = torch.cat((h_pooled, attn_pooled), 1)

        d = self.dropout(skip_connection)
        y = self.fc1(d)
        return y

"""
LSTM - CNN model

We apply a CNN as described earlier to the LSTM output

"""

class LSTMCNN(nn.Module):

    def __init__(self, TEXT, LABEL, args):
        super(LSTMCNN, self).__init__()
        self.vocab_size = len(TEXT.vocab)
        self.label_size = len(LABEL.vocab)
        self.embed_dim = args['embed_size']

        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        if not args['no_pretrained_vectors']:
            self.embed.weight = nn.Parameter(TEXT.vocab.vectors)

        # biderectional LSTM layer
        self.lstm = nn.LSTM(300, 64, 1, batch_first=True, dropout=0.5, bidirectional=True)
        self.c0 = nn.Parameter(torch.zeros((2, 1, 64)))
        self.h0 = nn.Parameter(torch.zeros((2, 1, 64)))

        # cnn
        self.conv1 = nn.Conv1d(128, 64, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(128, 64, 4, padding=2, stride=1)
        self.conv3 = nn.Conv1d(128, 64, 5, padding=2, stride=1)

        self.fc1 = nn.Linear(192, self.label_size)

    def forward(self, batch, training=False):
        h = self.h0.expand((-1, batch.size(0), -1)).contiguous()
        c = self.h0.expand((-1, batch.size(0), -1)).contiguous()


        H, _ = self.lstm(self.embed(batch), (h, c))
        a = F.relu(H)

        a = a.transpose(1,2)

        conv1 = F.relu(self.conv1(a))
        conv2 = F.relu(self.conv2(a))
        conv3 = F.relu(self.conv3(a))

        z1 = F.max_pool1d(conv1, conv1.size(2)).view(batch.size(0), -1)
        z2 = F.max_pool1d(conv2, conv2.size(2)).view(batch.size(0), -1)
        z3 = F.max_pool1d(conv3, conv3.size(2)).view(batch.size(0), -1)

        z = torch.cat((z1, z2, z3), dim=1)

        d = F.dropout(z, 0.5, training)
        y = self.fc1(d).squeeze()

        return y


class Transformer(nn.Module):
    def __init__(self, TEXT, LABEL, args):
        super(Transformer, self).__init__()
        self.vocab_size = len(TEXT.vocab)
        self.label_size = len(LABEL.vocab)
        self.embed_dim = args['embed_size']
        self.layers = args['num_layers']
        self.hidden_size = args['hidden_size']

        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        if not args['no_pretrained_vectors']:
            self.embed.weight = nn.Parameter(TEXT.vocab.vectors)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim,
            nhead=args['num_heads'], dropout=args['dropout'], dim_feedforward=self.hidden_size)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.layers)

        self.fc1 = nn.Linear(self.embed_dim, self.label_size)

        self.dropout = nn.Dropout(args['dropout'])

    def forward(self, batch):
        H = self.transformer(self.embed(batch))
        z = F.max_pool1d(H.transpose(1,2), H.size(1)).squeeze()
        d = self.dropout(z)
        y = self.fc1(d)
        return y

class Ensemble(nn.Module):
    def __init__(self, models, args):
        super(Ensemble, self).__init__()
        self.models = models
        self.label_size = len(args['LABEL'].vocab)
        self.ensemble_mode = args['ensemble']

        for i in range(len(self.models)):
            self.models[i].eval()

    def forward(self, batch):
        if self.ensemble_mode == 'average':
            scores = 0
            for model in self.models:
                scores += model(batch)
            scores /= len(self.models)
            return scores
        else:
            results = torch.zeros(batch.shape[0], self.label_size)
            predictions = []
            for model in self.models:
                predictions.append(utils.predict(model, batch))
            for i in range(batch.shape[0]):
                count = Counter()
                for idx in range(len(self.models)):
                    count[predictions[idx][i]]+=1
                results[i][count.most_common(1)[0][0]] = 1
            return results



model_dict = {
    'cbow': CBOW,
    'cnn': CNN,
    'cnn2': CNN2,
    'lstm': LSTM,
    'lstm_cnn': LSTMCNN,
    'lstm_att': LSTMSelfAttention,
    'transformer': Transformer
}

optimizer_dict = {
    'adam': torch.optim.Adam,
    'adamax': torch.optim.Adamax
}

loss = nn.CrossEntropyLoss()