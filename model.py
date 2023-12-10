import torch
import torch.nn as nn

from transformers import BertModel

# define a self attention layer that used below
class Attention(nn.Module):
  def __init__(self, hidden_dim):
    super(Attention, self).__init__()
    # the attention linear layer which transforms the input data to the hidden space
    self.attn = nn.Linear(hidden_dim * 4, hidden_dim * 2)
    # the linear layer that calculates the attention scores
    self.v = nn.Linear(hidden_dim * 2, 1, bias=False)

  def forward(self, hidden_state, inputs):
    # hidden_state.shape = (B, H)
    # inputs.shape = (B, S, H)

    seq_len = inputs.shape[1]
    # print(hidden_state.shape, inputs.shape)
    # expand hidden_state to (B, S, H)
    hidden_repeated = hidden_state.unsqueeze(1).repeat(1, seq_len, 1)
    # print(hidden_repeated.shape)
    # calculate attention weights
    attn_weights = torch.tanh(self.attn(torch.cat((hidden_repeated, inputs), dim=2)))
    # calculate attention scores in (B, S)
    scores = self.v(attn_weights).squeeze(2)
    # aplly softmax to get valid probabilities
    probs = nn.functional.softmax(scores, dim=1)
    # calculated weighted outputs
    out = probs.unsqueeze(1).bmm(inputs).squeeze(1)
    return out

class BiLSTMWithAttention(nn.Module):
  def __init__(self, embedding_matrix, hidden_dim, output_dim, n_layers, dropout):
    super(BiLSTMWithAttention, self).__init__()
    self.num_layers = n_layers
    self.hidden_dim = hidden_dim
    self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=False)
    self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_dim, num_layers=n_layers, bidirectional=True, batch_first=True, dropout=dropout)
    self.attention = Attention(hidden_dim)
    self.fc = nn.Linear(hidden_dim*2, output_dim)
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, x, hidden):
    # initailize hidden state
    x = self.embedding(x)
    lstm_out, hidden_state = self.lstm(x, hidden)
    hidden_state = torch.cat([hidden_state[-1], hidden_state[-2]], dim=-1)
    attention_out = self.attention(hidden_state[0], lstm_out)
    out = self.softmax(self.fc(attention_out))
    return out

  def init_hidden_state(self, batch_size, device):
      h0 = torch.zeros(self.num_layers*2, batch_size, self.hidden_dim).to(device)
      c0 = torch.zeros(self.num_layers*2, batch_size, self.hidden_dim).to(device)
      return (h0, c0)

class Bert_Classifier(nn.Module):
  def __init__(self, freeze_bert=False):
    super(Bert_Classifier, self).__init__()
    # specify hidden dimension of BERT and the classifier, and number of labels
    n_input = 768
    n_hidden = 50
    n_output = 5
    # initialize BERT model
    self.bert = BertModel.from_pretrained('bert-base-uncased')
    # define a simple classifier
    self.classifier = nn.Sequential(
        nn.Linear(n_input, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_output)
    )
    # freeze BERT parameters if specified
    if freeze_bert:
      for param in self.bert.parameters():
        param.requires_grad = False
  def forward(self, input_ids, attention_mask):
    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    # extract the last hidden state of the `[CLS]` token from the BERT output (useful for classification tasks)
    last_hidden_state_cls = outputs[0][:, 0, :]
    # feed the extracted hidden state to the classifier to compute logits
    logits = self.classifier(last_hidden_state_cls)
    return logits