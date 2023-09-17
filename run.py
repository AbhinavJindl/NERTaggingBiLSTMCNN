import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.optim import lr_scheduler
torch.manual_seed(25)

# global variables
train_file = './data/train'
dev_file = './data/dev'
test_file = './data/test'
device = "cuda"
task1_model_path = './blstm1.pt' 
task2_model_path = './blstm2.pt' 
task3_model_path = 'blstm3.pt' 


"""
Helper functions to read data files and create sentences and tags
"""
def read_train_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        f.close()

    sentences = []
    sentence_tags = []
    current_sentence = []
    current_tags = []
    for line in lines:
        line = line.strip()
        if line == '':
            sentences.append(current_sentence)
            sentence_tags.append(current_tags)
            current_sentence = []
            current_tags = []
            continue
        [index, word, tag] = line.split(' ')
        current_sentence.append(word)
        current_tags.append(tag)
    sentences.append(current_sentence)
    sentence_tags.append(current_tags)
    return sentences, sentence_tags

def read_test_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        f.close()
        
    sentences = []
    current_sentence = []
    for line in lines:
        line = line.strip()
        if line == '':
            sentences.append(current_sentence)
            current_sentence = []
            continue
        [index, word] = line.split(' ')
        current_sentence.append(word)
    sentences.append(current_sentence)
    return sentences    

train_sentences, train_sentence_tags = read_train_data(train_file)        
dev_sentences, dev_sentence_tags = read_train_data(dev_file)
test_sentences = read_test_data(test_file)

"""
Dataset and train helper methods
"""
# Dataset for DataLoader
class NERDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.lengths = [len(x) for x in self.X]
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.lengths[idx]

def pad_collate(batch):
    (xx, yy, lengths) = zip(*batch)
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0).type(torch.LongTensor).to(device)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=-1).type(torch.LongTensor).to(device)
    return xx_pad, yy_pad, lengths   

# Train function
def train(model, train_data, optimizer, criterion):
    model.train()
    for i, batch in enumerate(train_data):
        sentences, tags, lengths = batch
        optimizer.zero_grad()
        output = model(sentences, lengths)
        loss = criterion(output.view(-1, output.shape[-1]), tags.view(-1))
        loss.backward()
        optimizer.step()

"""
Prediction and Evaluation Helper Methods
"""

def predict(model, eval_data, batch_size):
    predicted_tags = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            batch = eval_data[i:i+batch_size]
            lengths = [len(x) for x in batch]
            sentences = pad_sequence(batch, batch_first=True, padding_value=0).type(torch.LongTensor).to(device)
            output = model(sentences, lengths)
            preds = output.argmax(dim=-1)
            for idx, sentence in enumerate(batch):
                pad_index = len(sentence)
                predicted_tags.append(preds[idx, 0: pad_index])
    return predicted_tags

def get_predicted_sentence_tags(preds, labels_vocab):
    itos = labels_vocab
    sentence_tags = []
    for i in range(len(preds)):
        current_sentence_tags = []
        j = 0
        while j<len(preds[i]):
            current_sentence_tags.append(itos[int(preds[i][j])])
            j += 1
        sentence_tags.append(current_sentence_tags)
    return sentence_tags

def write_gold_dev_output(filename, sentences, sentence_tags, predicted_tags):
    with open(filename, 'w') as f:
        s = ""
        for i in range(len(sentences)):
            assert len(sentences[i]) == len(predicted_tags[i])
            if i != 0:
                s += '\n'
            for j in range(len(sentences[i])):
                s += "{} {} {} {}\n".format(j+1, sentences[i][j], sentence_tags[i][j], predicted_tags[i][j])
        f.write(s)
        f.close()

def write_test_output(filename, sentences, predicted_tags):
    with open(filename, 'w') as f:
        s = ""
        for i in range(len(sentences)):
            assert len(sentences[i]) == len(predicted_tags[i])
            if i != 0:
                s += '\n'
            for j in range(len(sentences[i])):
                s += "{} {} {}\n".format(j+1, sentences[i][j], predicted_tags[i][j])
        f.write(s)
        f.close()
        
def calculate_score(true_tags, predicted_tags):
    true_tags = list(np.concatenate(true_tags))
    predicted_tags = list(np.concatenate(predicted_tags))
    total = len(predicted_tags)
    matched = 0
    assert(len(predicted_tags) == len(true_tags))
    for i in range(len(predicted_tags)):
        if predicted_tags[i] == true_tags[i]:
            matched += 1
    return matched/total


"""
Task 1: Simple BLSTM
"""

# Create numerical sentences with vocab
def create_vocab(sentences, vocab, is_X):
    if vocab is None:
        word_frequencies = {}
        for sentence in sentences:
            for word in sentence:
                if word not in word_frequencies:
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] = word_frequencies[word] + 1
                        
        threshold = 2
        keys = list(word_frequencies.keys())
        for key in keys:
            if word_frequencies[key] <= threshold:
                del word_frequencies[key]
        
        vocab =  list(word_frequencies.keys())
        if is_X:
            vocab = ["<pad>", "<unk>"] + vocab

    word_to_index = {}
    for idx, word in enumerate(vocab):
        word_to_index[word] = idx

    numerical_sentences = []
    for sentence in sentences:
        temp_vec = []
        for word in sentence:
            if word not in word_to_index:
                temp_vec.append(word_to_index["<unk>"])
            else:
                temp_vec.append(word_to_index[word])

        numerical_sentences.append(torch.tensor(temp_vec).to(torch.int))

    return vocab, numerical_sentences


class NERModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, linear_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim*2, linear_dim)
        self.elu = nn.ELU()
        self.out = nn.Linear(linear_dim, output_dim)
        self.dropout = nn.Dropout(0.33)

    def forward(self, x, lengths):
        # text shape: (batch_size, seq_length)
        embedded = self.embedding(x)  # embedded shape: (batch_size, seq_length, embedding_dim)
        lstm_output = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        lstm_output, _ = self.lstm(lstm_output) # lstm_output shape: (batch_size, seq_length, hidden_dim*2)
        lstm_output, _ = pad_packed_sequence(lstm_output, batch_first=True)
        lstm_output = self.dropout(lstm_output)
        linear_output = self.fc(lstm_output)  # linear_output shape: (batch_size, seq_length, hidden_dim)
        elu_output = self.elu(linear_output)  # elu_output shape: (batch_size, seq_length, hidden_dim)
        output = self.out(elu_output)  # tag_space shape: (batch_size, seq_length, output_dim)
        return output  
    

def run_simple_blstm_training():
    # Define hyperparameters
    vocab_size = len(vocab)
    embedding_dim = 100
    num_layers = 1
    hidden_dim = 256
    linear_dim=128
    output_dim = len(labels_vocab)
    lr = 1
    batch_size = 64
    num_epochs = 40
    train_dataset = NERDataset(train_X, train_Y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    val_dataset = NERDataset(dev_X, dev_Y)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)

    # Create model and optimizer
    model = NERModel(vocab_size, embedding_dim, hidden_dim, num_layers, linear_dim, output_dim)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    for epoch in range(num_epochs):
        train(model, train_loader, optimizer, criterion)
        predicted_train_tags = get_predicted_sentence_tags(predict(model, train_X, batch_size), labels_vocab)
        train_accuracy = calculate_score(train_sentence_tags, predicted_train_tags)
        predicted_dev_tags = get_predicted_sentence_tags(predict(model, dev_X, batch_size), labels_vocab)
        val_accuracy = calculate_score(dev_sentence_tags, predicted_dev_tags)
        print(f'Epoch {epoch+1}: Train Accuracy={train_accuracy:.4f}')
        print(f'Epoch {epoch+1}: Validation Accuracy={val_accuracy:.4f}')
        scheduler.step()
    
    torch.save(model, task1_model_path)

def run_simple_blstm_predictions():
    model = torch.load(task1_model_path)
    model.eval()
    # Evaluate Model and Create Prediction Files
    predicted_dev_tags = get_predicted_sentence_tags(predict(model, dev_X, 128), labels_vocab)
    write_test_output('./dev1.out', dev_sentences, predicted_dev_tags)
    predicted_test_tags = get_predicted_sentence_tags(predict(model, test_X, 128), labels_vocab)
    write_test_output('./test1.out', test_sentences, predicted_test_tags)
    write_gold_dev_output('./gold1.out', dev_sentences, dev_sentence_tags, predicted_dev_tags)
    # print(calculate_score(dev_sentence_tags, predicted_dev_tags))
    # !perl conll03eval.txt < dev.out


"""
Task 2: Glove Word Embeddings
"""
def create_glove_vocab():
    with open('./glove.6B.100d.txt', encoding='utf-8') as f: 
        lines = f.readlines()
        word_to_vec = {}
        word_to_vec["<pad>"] = np.concatenate((np.random.rand(100), np.zeros([1])))
        unk_vec = np.random.rand(100)
        word_to_vec["<unk>"] = np.concatenate((unk_vec, np.zeros([1])))
        word_to_vec["<UNK>"] = np.concatenate((unk_vec, np.ones([1])))
        for line in lines:
            tokens = line.split()
            vec = np.array([float(x) for x in tokens[1:]])
            s = str(tokens[0])
            lower_vec = np.concatenate((vec, np.zeros([1])))
            upper_vec = np.concatenate((vec, np.ones([1])))
            word_to_vec[s] = lower_vec
            word_to_vec[s.upper()] = upper_vec

    vocab = list(word_to_vec.keys())
    assert(vocab.index("<pad>") == 0)
    vocab_size = len(vocab)
    embeddings_weights = np.zeros([vocab_size, 101])
    for idx, key in enumerate(vocab):
        embeddings_weights[idx] = word_to_vec[key]

    return vocab, vocab_size, embeddings_weights


def create_numerical_vectors(sentences, vocab):
    vocab_to_index = {}
    for idx, v in enumerate(vocab):
        vocab_to_index[v] = idx
    numerical_sentences = []
    vocab_set = set(vocab)
    for sentence in sentences:
        vector = []
        for token in sentence:
            if token.lower() in vocab_set:
                if token.lower() == token:
                    vector.append(vocab_to_index[token])
                else:
                    vector.append(vocab_to_index[token.upper()])
            else:
                if token.lower() == token:
                    vector.append(vocab_to_index["<unk>"])
                else:
                    vector.append(vocab_to_index["<unk>".upper()])
        numerical_sentences.append(torch.tensor(vector).to(torch.int))
    return numerical_sentences

class GloveNERModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_weight, hidden_dim, num_layers, linear_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim*2, linear_dim)
        self.elu = nn.ELU()
        self.out = nn.Linear(linear_dim, output_dim)
        self.dropout = nn.Dropout(0.33)

    def forward(self, x, lengths):
        # text shape: (batch_size, seq_length)
        embedded = self.embedding(x)  # embedded shape: (batch_size, seq_length, embedding_dim)
        lstm_output = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        lstm_output, _ = self.lstm(lstm_output) # lstm_output shape: (batch_size, seq_length, hidden_dim*2)
        lstm_output, _ = pad_packed_sequence(lstm_output, batch_first=True)
        lstm_output = self.dropout(lstm_output)
        linear_output = self.fc(lstm_output)  # linear_output shape: (batch_size, seq_length, hidden_dim)
        elu_output = self.elu(linear_output)  # elu_output shape: (batch_size, seq_length, hidden_dim)
        output = self.out(elu_output)  # tag_space shape: (batch_size, seq_length, output_dim)
        return output 
    
def run_glove_training():
    # hyperparameters
    vocab_size = len(vocab)
    embedding_dim = 101
    num_layers = 1
    hidden_dim = 256
    linear_dim=128
    output_dim = len(labels_vocab)
    lr = 1
    batch_size = 64
    num_epochs = 40
    train_dataset = NERDataset(train_X, train_Y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
    val_dataset = NERDataset(dev_X, dev_Y)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)

    model = GloveNERModel(vocab_size, embedding_dim, embeddings_weights, hidden_dim, num_layers, linear_dim, output_dim)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    for epoch in range(num_epochs):
        train(model, train_loader, optimizer, criterion)
        predicted_train_tags = get_predicted_sentence_tags(predict(model, train_X, batch_size), labels_vocab)
        train_accuracy = calculate_score(train_sentence_tags, predicted_train_tags)
        predicted_dev_tags = get_predicted_sentence_tags(predict(model, dev_X, batch_size), labels_vocab)
        val_accuracy = calculate_score(dev_sentence_tags, predicted_dev_tags)
        print(f'Epoch {epoch+1}: Train Accuracy={train_accuracy:.4f}')
        print(f'Epoch {epoch+1}: Validation Accuracy={val_accuracy:.4f}')
        scheduler.step()
    
    # save model to file
    torch.save(model, task2_model_path)

def run_glove_prediction():
    model = torch.load(task2_model_path)
    model.eval()
    # print(calculate_score(dev_sentence_tags, predicted_dev_tags))
    predicted_dev_tags = get_predicted_sentence_tags(predict(model, dev_X, 128), labels_vocab)
    write_test_output('./dev2.out', dev_sentences, predicted_dev_tags)
    predicted_test_tags = get_predicted_sentence_tags(predict(model, test_X, 128), labels_vocab)
    write_test_output('./test2.out', test_sentences, predicted_test_tags)
    write_gold_dev_output('./gold2.out', dev_sentences, dev_sentence_tags, predicted_dev_tags)
    # !perl conll03eval.txt < dev.out


"""
Bonus Task
"""

def create_char_vocab(sentences, vocab, max_word_length):
    if max_word_length is None:
        max_word_length = 0
        for sentence in sentences:
            for word in sentence:
                if len(word) > max_word_length:
                    max_word_length = len(word)
                
    if vocab is None:
        char_frequencies = {}
        for sentence in sentences:
            for word in sentence:
                for char in word:
                    if char not in char_frequencies:
                        char_frequencies[char] = 1
                    else:
                        char_frequencies[char] = char_frequencies[char] + 1
                        
        threshold = 2
        keys = list(char_frequencies.keys())
        for key in keys:
            if char_frequencies[key] <= threshold:
                del char_frequencies[key]
        
        vocab =  list(char_frequencies.keys())
        vocab = ["<pad>", "<unk>"] + vocab

    char_to_index = {}
    for idx, char in enumerate(vocab):
        char_to_index[char] = idx

    numerical_sentences = []
    for sentence in sentences:
        sentence_vec = []
        for word in sentence:
            word_tensor = torch.zeros([max_word_length])
            word_vec = []
            for char in word:
                if char not in char_to_index:
                    word_vec.append(char_to_index["<unk>"])
                else:
                    word_vec.append(char_to_index[char])
            word_tensor[0:len(word_vec)] = torch.tensor(word_vec).to(torch.int)
            sentence_vec.append(word_tensor)
        sentence_vec = pad_sequence(sentence_vec, batch_first=True, padding_value=0).type(torch.int).to(device)
        numerical_sentences.append(sentence_vec)
    return vocab, numerical_sentences, max_word_length


class CharCNN(nn.Module):
    def __init__(self, char_vocab_size, char_embedding_dim, output_dim):
        super().__init__()
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(char_embedding_dim, output_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1)
        self.elu = nn.ELU()
        
    def forward(self, x):
        # x is a tensor of shape (batch_size, seq_len, max_word_length)
        batch_size, seq_len, max_word_length = x.shape
        x = x.view(-1, max_word_length)  # reshape to (batch_size * seq_len, max_word_length)
        x = self.char_embedding(x)  # (batch_size * seq_len, max_word_length, char_embedding_dim)
        x = x.transpose(1, 2)  # (batch_size * seq_len, char_embedding_dim, max_word_length)
        x = self.elu(self.conv1(x))  # (batch_size * seq_len, output_dim, max_word_length)
        x = self.elu(self.conv2(x))  # (batch_size * seq_len, output_dim, max_word_length)
        x, _ = torch.max(x, dim=2)  # (batch_size * seq_len, output_dim)
        x = x.view(batch_size, seq_len, -1)  # reshape to (batch_size, seq_len, output_dim)
        return x

class CharCNNLSTM(nn.Module):
    def __init__(self, word_vocab_size, word_embedding_dim, pretrained_weight, char_vocab_size, char_embedding_dim, char_output_dim, hidden_dim, num_layers, linear_dim, output_dim):
        super().__init__()
        self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim, padding_idx=0)
        self.word_embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.char_cnn = CharCNN(char_vocab_size, char_embedding_dim, char_output_dim)
        self.lstm = nn.LSTM(word_embedding_dim + char_output_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim*2, linear_dim)
        self.elu = nn.ELU()
        self.out = nn.Linear(linear_dim, output_dim)
        self.dropout = nn.Dropout(0.33)
        
    def forward(self, chars, words, lengths):
        # words is a tensor of shape (batch_size, seq_len)
        # chars is a tensor of shape (batch_size, seq_len, max_word_length)
        word_embedded = self.word_embedding(words)  # (batch_size, seq_len, word_embedding_dim)
        char_embedded = self.char_cnn(chars)  # (batch_size, seq_len, char_output_dim)
        combined_embedded = torch.cat((word_embedded, char_embedded), dim=2)  # (batch_size, seq_len, word_embedding_dim + char_output_dim)
        lstm_output = pack_padded_sequence(combined_embedded, lengths, batch_first=True, enforce_sorted=False)
        lstm_output, _ = self.lstm(lstm_output) # lstm_output shape: (batch_size, seq_length, hidden_dim*2)
        lstm_output, _ = pad_packed_sequence(lstm_output, batch_first=True)
        lstm_output = self.dropout(lstm_output)
        linear_output = self.fc(lstm_output)  # linear_output shape: (batch_size, seq_length, hidden_dim)
        elu_output = self.elu(linear_output)  # elu_output shape: (batch_size, seq_length, hidden_dim)
        elu_output = self.dropout(elu_output)
        output = self.out(elu_output)
        return output
        

# Dataset for DataLoader
class CNNNERDataset(Dataset):
    def __init__(self, char_X, X, y):
        self.char_X = char_X
        self.X = X
        self.lengths = [len(x) for x in self.X]
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.char_X[idx], self.X[idx], self.y[idx], self.lengths[idx]

def cnn_pad_collate(batch):
    (char_xx, xx, yy, lengths) = zip(*batch)
    char_xx = pad_sequence(char_xx, batch_first=True, padding_value=0).type(torch.LongTensor).to(device)
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0).type(torch.LongTensor).to(device)
    yy_pad = pad_sequence(yy, batch_first=True, padding_value=-1).type(torch.LongTensor).to(device)
    
    return char_xx, xx_pad, yy_pad, lengths   

# Train function
def cnn_train(model, train_data, optimizer, criterion):
    model.train()
    for i, batch in enumerate(train_data):
        chars, sentences, tags, lengths = batch
        optimizer.zero_grad()
        output = model(chars, sentences, lengths)
        loss = criterion(output.view(-1, output.shape[-1]), tags.view(-1))
        loss.backward()
        optimizer.step()
        
def cnn_predict(model, eval_data, char_eval_data, batch_size):
    predicted_tags = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            batch = eval_data[i:i+batch_size]
            char_batch = char_eval_data[i:i+batch_size]
            lengths = [len(x) for x in batch]
            sentences = pad_sequence(batch, batch_first=True, padding_value=0).type(torch.LongTensor).to(device)
            char_sentences = pad_sequence(char_batch, batch_first=True, padding_value=0).type(torch.LongTensor).to(device)
            output = model(char_sentences, sentences, lengths)
            preds = output.argmax(dim=-1)
            for idx, sentence in enumerate(batch):
                pad_index = len(sentence)
                predicted_tags.append(preds[idx, 0: pad_index])
    return predicted_tags


def run_cnn_lstm_training():
    vocab, vocab_size, embeddings_weights = create_glove_vocab()
    train_X = create_numerical_vectors(train_sentences, vocab)
    labels_vocab, train_Y = create_vocab(train_sentence_tags, vocab=None, is_X=False)
    dev_X = create_numerical_vectors(dev_sentences, vocab)
    dev_labels_vocab, dev_Y = create_vocab(dev_sentence_tags, vocab=labels_vocab, is_X=False)
    test_X = create_numerical_vectors(test_sentences, vocab)

    char_vocab, char_train_X, max_word_length = create_char_vocab(train_sentences, None, None)
    _, char_dev_X, max_word_length = create_char_vocab(dev_sentences, char_vocab, max_word_length)
    _, char_test_X, max_word_length = create_char_vocab(test_sentences, char_vocab, max_word_length)

    word_vocab_size = len(vocab)
    char_vocab_size = len(char_vocab)
    word_embedding_dim = 101
    char_embedding_dim = 60
    char_output_dim = 30
    num_layers = 1
    hidden_dim = 256
    linear_dim = 128
    output_dim = len(labels_vocab)
    lr = 1
    batch_size = 64
    num_epochs = 50
    train_dataset = CNNNERDataset(char_train_X, train_X, train_Y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=cnn_pad_collate)
    val_dataset = CNNNERDataset(char_train_X, dev_X, dev_Y)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=cnn_pad_collate)
    
    
    model = CharCNNLSTM(
        word_vocab_size, 
        word_embedding_dim, 
        embeddings_weights, 
        char_vocab_size, 
        char_embedding_dim, 
        char_output_dim, 
        hidden_dim, 
        num_layers, 
        linear_dim, 
        output_dim)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5, 30], gamma=0.1)

    for epoch in range(num_epochs):
        cnn_train(model, train_loader, optimizer, criterion)
        predicted_train_tags = get_predicted_sentence_tags(cnn_predict(model, train_X, char_train_X, batch_size), labels_vocab)
        train_accuracy = calculate_score(train_sentence_tags, predicted_train_tags)
        predicted_dev_tags = get_predicted_sentence_tags(cnn_predict(model, dev_X, char_dev_X, batch_size), labels_vocab)
        val_accuracy = calculate_score(dev_sentence_tags, predicted_dev_tags)
        print(f'Epoch {epoch+1}: Train Accuracy={train_accuracy:.4f}')
        print(f'Epoch {epoch+1}: Validation Accuracy={val_accuracy:.4f}')

    # save model to file
    torch.save(model, task3_model_path)
        
    return model


def run_cnn_lstm_prediction():
    vocab, vocab_size, embeddings_weights = create_glove_vocab()
    train_X = create_numerical_vectors(train_sentences, vocab)
    labels_vocab, train_Y = create_vocab(train_sentence_tags, vocab=None, is_X=False)
    dev_X = create_numerical_vectors(dev_sentences, vocab)
    dev_labels_vocab, dev_Y = create_vocab(dev_sentence_tags, vocab=labels_vocab, is_X=False)
    test_X = create_numerical_vectors(test_sentences, vocab)

    char_vocab, char_train_X, max_word_length = create_char_vocab(train_sentences, None, None)
    _, char_dev_X, max_word_length = create_char_vocab(dev_sentences, char_vocab, max_word_length)
    _, char_test_X, max_word_length = create_char_vocab(test_sentences, char_vocab, max_word_length)

    model = torch.load(task3_model_path)
    model.eval()

    predicted_dev_tags = get_predicted_sentence_tags(cnn_predict(model, dev_X, char_dev_X, 128), labels_vocab)
    write_test_output('./dev3.out', dev_sentences, predicted_dev_tags)
    write_gold_dev_output('./gold3.out', dev_sentences, dev_sentence_tags, predicted_dev_tags)

    predicted_test_tags = get_predicted_sentence_tags(cnn_predict(model, test_X, char_test_X, 128), labels_vocab)
    write_test_output('./pred', test_sentences, predicted_test_tags)
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", help="train models", action="store_true")
    parser.add_argument("-p", "--predict", help="run prediction", action="store_true")
    args = parser.parse_args()
    if args.train == args.predict:
        raise Exception("One of --predict or --train should be passed.")
    
    if args.train:
        print ("Running Task 1 Training......")
        vocab, train_X = create_vocab(train_sentences, vocab=None, is_X=True)
        labels_vocab, train_Y = create_vocab(train_sentence_tags, vocab=None, is_X=False)
        dev_vocab, dev_X = create_vocab(dev_sentences, vocab=vocab, is_X=True)
        dev_labels_vocab, dev_Y = create_vocab(dev_sentence_tags, vocab=labels_vocab, is_X=False)
        test_vocab, test_X = create_vocab(test_sentences, vocab=vocab, is_X=True)
        run_simple_blstm_training()

        print ("Running Task 2 Training......")
        vocab, vocab_size, embeddings_weights = create_glove_vocab()
        train_X = create_numerical_vectors(train_sentences, vocab)
        labels_vocab, train_Y = create_vocab(train_sentence_tags, vocab=None, is_X=False)
        dev_X = create_numerical_vectors(dev_sentences, vocab)
        dev_labels_vocab, dev_Y = create_vocab(dev_sentence_tags, vocab=labels_vocab, is_X=False)
        test_X = create_numerical_vectors(test_sentences, vocab)
        run_glove_training()

        print ("Running Bonus Task Training......")
        run_cnn_lstm_training()


    if args.predict:
        print ("Running Task 1 Predictions.......")
        vocab, train_X = create_vocab(train_sentences, vocab=None, is_X=True)
        labels_vocab, train_Y = create_vocab(train_sentence_tags, vocab=None, is_X=False)
        dev_vocab, dev_X = create_vocab(dev_sentences, vocab=vocab, is_X=True)
        dev_labels_vocab, dev_Y = create_vocab(dev_sentence_tags, vocab=labels_vocab, is_X=False)
        test_vocab, test_X = create_vocab(test_sentences, vocab=vocab, is_X=True)
        run_simple_blstm_predictions()

        print ("Running Task 2 Predictions.......")
        vocab, vocab_size, embeddings_weights = create_glove_vocab()
        train_X = create_numerical_vectors(train_sentences, vocab)
        labels_vocab, train_Y = create_vocab(train_sentence_tags, vocab=None, is_X=False)
        dev_X = create_numerical_vectors(dev_sentences, vocab)
        dev_labels_vocab, dev_Y = create_vocab(dev_sentence_tags, vocab=labels_vocab, is_X=False)
        test_X = create_numerical_vectors(test_sentences, vocab)
        run_glove_prediction()

        print ("Running Bonus Task Predictions.......")
        run_cnn_lstm_prediction()
