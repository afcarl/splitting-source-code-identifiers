import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import f1_score

import random
import json
import string
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str, default='data/train_examples.json')
parser.add_argument('--test_file', type=str, default='data/test_examples.json')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--unidirectional', action='store_true', default=False)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--chunks', type=int, default=10)
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--seed', type=int, default=1234)
args = parser.parse_args()

print(args)

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

vocab = set(string.ascii_lowercase + string.digits)

vocab.add('<pad>')
vocab.add('<unk>')

char2idx = {c:i for i, c in enumerate(vocab)}
idx2char = {i:c for i, c in enumerate(vocab)}

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
            bidirectional, n_layers, one_hot=False):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim,
                bidirectional=bidirectional, num_layers=n_layers)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):

        #x = [bsz, seq len]

        e = self.embedding(x)

        #e = [bsz, seq len, emb dim]

        H, (_, _) = self.rnn(e.permute(1, 0, 2))

        #H = [seq len, bsz, n directions * hid size]

        o = self.fc(H.permute(1, 0, 2))

        #o = [bsz, seq len, output dim]

        return o


model = LSTMModel(len(vocab), args.embedding_dim, args.hidden_dim, 1, not
        args.unidirectional, args.n_layers)

def file_iterator(path, batch_size, chunks):
    """
    Too many examples to store in memory, so this will load batch_size * chunks
    examples into memory and then shuffle and yield them
    """

    examples = []

    with open(path, 'r') as f:

        for line in f:

            examples.append(json.loads(line))

            if len(examples) >= (batch_size * chunks):
                random.shuffle(examples)
                yield examples
                examples = []

    yield examples

def pad(examples, pad_token):
    """
    Get list of list of examples, all different length and pad to the same
    length
    """

    max_len = max([len(e) for e in examples])

    for i, e in enumerate(examples):

        while len(examples[i]) < max_len:
            examples[i].append(pad_token)

    return examples

def tensor_iterator(examples, batch_size):
    """
    Get batch_size * chunks examples where each is a dict and yields a batch of
    numericalized tensor inputs and outputs
    """

    x = [e['identifier'] for e in examples]
    y = [e['target'] for e in examples]

    x = [[char2idx[char] for char in e] for e in x]

    n_batches = len(examples) // batch_size

    for i in range(n_batches):

        batch_x = pad(x[i*batch_size:(i+1)*batch_size],
                pad_token=char2idx['<pad>'])
        batch_y = pad(y[i*batch_size:(i+1)*batch_size],
                pad_token=0)

        yield torch.LongTensor(batch_x), torch.FloatTensor(batch_y)

def calculate_f1(predictions, targets):

    predictions = torch.round(torch.sigmoid(predictions)).tolist()
    targets = targets.tolist()

    binary_f1 = sum([f1_score(t, p, average='binary') for t, p in zip(targets, predictions)])
    micro_f1 = sum([f1_score(t, p, average='micro') for t, p in zip(targets, predictions)])
    macro_f1 = sum([f1_score(t, p, average='macro') for t, p in zip(targets, predictions)])
    weighted_f1 = sum([f1_score(t, p, average='weighted') for t, p in zip(targets, predictions)])
    
    return binary_f1, micro_f1, macro_f1, weighted_f1

def train(model, path, device, criterion, optimizer):

    model.train()

    epoch_loss = 0
    epoch_binary_f1 = 0
    epoch_micro_f1 = 0
    epoch_macro_f1 = 0
    epoch_weighted_f1 = 0
    epoch_batches = 0
    epoch_examples = 0

    for i, examples in enumerate(file_iterator(path, args.batch_size, args.chunks), start=1):

        for x, y in tensor_iterator(examples, args.batch_size):

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            predictions = model(x).squeeze(2)

            mask = (x != char2idx['<pad>']).float()

            del x

            loss = criterion(predictions, y)

            binary_f1, micro_f1, macro_f1, weighted_f1 = calculate_f1(predictions, y)

            del predictions
            del y

            masked_loss = loss * mask

            loss = masked_loss.sum() / mask.shape[0]

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            epoch_binary_f1 += binary_f1
            epoch_micro_f1 += micro_f1
            epoch_macro_f1 += macro_f1
            epoch_weighted_f1 += weighted_f1
            epoch_examples += mask.shape[0]
            epoch_batches += 1

        if i % args.print_every == 0:
            print(f'Examples: {i*args.batch_size*args.chunks:08}, Avg Loss per Batch: {epoch_loss/epoch_batches:.3f}, Avg F1 per Example: {epoch_binary_f1/epoch_examples:.3f}, {epoch_micro_f1/epoch_examples:.3f}, {epoch_macro_f1/epoch_examples:.3f}, {epoch_weighted_f1/epoch_examples:.3f}')

    return epoch_loss / epoch_batches, epoch_binary_f1 / epoch_examples, epoch_micro_f1 / epoch_examples, epoch_macro_f1 / epoch_examples, epoch_weighted_f1 / epoch_examples
    

def evaluate(model, path, device, criterion):

    model.eval()

    epoch_loss = 0
    epoch_binary_f1 = 0
    epoch_micro_f1 = 0
    epoch_macro_f1 = 0
    epoch_weighted_f1 = 0
    epoch_batches = 0
    epoch_examples = 0

    with torch.no_grad():

        for i, examples in enumerate(file_iterator(path, args.batch_size, args.chunks), start=1):

            for (x, y) in tensor_iterator(examples, args.batch_size):

                x = x.to(device)
                y = y.to(device)

                predictions = model(x).squeeze(2)

                mask = (x != char2idx['<pad>']).float()

                del x

                loss = criterion(predictions, y)

                binary_f1, micro_f1, macro_f1, weighted_f1 = calculate_f1(predictions, y)

                del predictions
                del y

                masked_loss = loss * mask

                loss = masked_loss.sum()

                epoch_loss += loss.item()
                epoch_binary_f1 += binary_f1
                epoch_micro_f1 += micro_f1
                epoch_macro_f1 += macro_f1
                epoch_weighted_f1 += weighted_f1
                epoch_batches += 1
                epoch_examples += mask.shape[0]

            if i % args.print_every == 0:
                print(f'Examples: {i*args.batch_size*args.chunks:08}, Avg Loss per Batch: {epoch_loss/epoch_batches:.3f}, Avg F1 per Example: {epoch_binary_f1/epoch_examples:.3f}, {epoch_micro_f1/epoch_examples:.3f}, {epoch_macro_f1/epoch_examples:.3f}, {epoch_weighted_f1/epoch_examples:.3f}')


    return epoch_loss / epoch_batches, epoch_binary_f1 / epoch_examples, epoch_micro_f1 / epoch_examples, epoch_macro_f1 / epoch_examples, epoch_weighted_f1 / epoch_examples


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

criterion = nn.BCEWithLogitsLoss(reduction='none')

best_test_loss = float('inf')

for epoch in range(1, args.epochs+1):

    train_loss, train_f1 = train(model, args.train_file, device, criterion, optimizer)
    test_loss, test_f1 = evaluate(model, args.test_file, device, criterion)

    print(f'Epoch {epoch}, Train Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f}')

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), 'model.pt')