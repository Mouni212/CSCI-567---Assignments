#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re


# In[2]:


train_path = r'data/train'

def process_input_from_file(file_path):
    sentences = []
    tags = []
    unique_words = []
    unique_tags = []
    with open(file_path, 'r') as file:
        data_blocks = file.read().strip().split('\n\n')
        
        for block in data_blocks:
            words = []
            word_tags = []
            for line in block.split('\n'):
                _, word, tag = line.split()
                words.append(word)
                word_tags.append(tag)
                if word not in unique_words:
                    unique_words.append(word)
                if tag not in unique_tags:
                    unique_tags.append(tag)
            sentences.append(' '.join(words))
            tags.append(' '.join(word_tags))
    
    return pd.DataFrame({
        'sentence': sentences,
        'tags': tags
    }), list(unique_words), list(unique_tags)

train_data_df, unique_words, unique_tags = process_input_from_file(train_path)
print(train_data_df.head())


# In[3]:


vocab_size = len(unique_words)
tag_size = len(unique_tags)
device="cuda"


# In[4]:


word_to_idx = {}
tag_to_idx = {}

for i in range(vocab_size):
  word_to_idx[unique_words[i]] = i+1
word_to_idx['<UNK>'] = 0

for i in range(tag_size):
  tag_to_idx[unique_tags[i]] = i


# In[5]:


import torch

max_size = train_data_df['sentence'].map(lambda x: len(x.split())).max()
def prepare_sequence(seq, to_ix):
    words = seq.split(" ")
    idxs = []
    for w in words:
        if w in to_ix:
            idxs.append(to_ix[w])
        else:
            idxs.append(0)
    return idxs


# In[6]:


train_data_df['words_numerical'] = train_data_df['sentence'].map(lambda seq: prepare_sequence(seq, word_to_idx))
train_data_df["tags_numerical"] = train_data_df['tags'].map(lambda tags: prepare_sequence(tags, tag_to_idx))


# In[7]:


from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

class NERDataset(Dataset):
    def __init__(self, sentences, tags = None):
        self.sentences = sentences
        self.tags = tags
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = torch.tensor(self.sentences.iloc[idx], dtype=torch.long)
        if self.tags is not None:
            tag = torch.tensor(self.tags.iloc[idx], dtype=torch.long)
            return sentence, tag, len(sentence)
        else:
            return sentence, len(sentence)

train_dataset = NERDataset(train_data_df["words_numerical"], train_data_df["tags_numerical"])

def pad_collate(batch):
    (sentences, tags, lengths) = zip(*batch)
    
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=word_to_idx['<UNK>'])
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=tag_to_idx['O'])  # Use the index of the 'O' tag or a suitable PAD token
    
    return sentences_padded, tags_padded, torch.tensor(lengths)
    
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=pad_collate)


# In[8]:


def get_case_pattern(word):
    if word.islower():
        return 'LOW'
    elif word.isupper():
        return 'UPP'
    elif word.istitle():
        return 'CAP'
    else:
        return 'MIX'


# In[10]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import gzip

glove_file_path = r"glove.6B.100d/glove.6B.100d.txt"

embedding_dim = 100

def augment_with_case_features(word, vector):
    # Create a basic case feature vector
    case_features = torch.tensor([int(word.islower()), int(word.isupper()), int(word.istitle())], dtype=torch.float)
    # Concatenate the original vector with the case features
    return torch.cat((vector, case_features))

def load_glove_embeddings(glove_file_path, include_case_features=False):
    embeddings_dict = {}
    with open(glove_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = torch.tensor([float(val) for val in values[1:]], dtype=torch.float)
            # if include_case_features:
                # vector = augment_with_case_features(word, vector)
            embeddings_dict[word] = vector
    return embeddings_dict

def create_embedding_matrix(word_to_idx, embeddings_dict, embedding_dim, include_case_features=False):
    # Adjust embedding dimension if case features are included
    additional_features_dim = 3 if include_case_features else 0
    vocab_size = len(word_to_idx)
    embedding_matrix = torch.zeros((vocab_size + 1, embedding_dim))

    for word, idx in word_to_idx.items():
        embedding_vector = embeddings_dict.get(word)
        if embedding_vector is None:
            embedding_vector = embeddings_dict.get(word.lower())
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector
        else:
            embedding_matrix[idx] = torch.tensor(np.random.normal(scale=0.28, size=(embedding_dim, )))
    
    return embedding_matrix

embedding_matrix = create_embedding_matrix(word_to_idx, load_glove_embeddings(glove_file_path), embedding_dim)


# In[11]:


import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self, vocab_size, embedding_size, hidden_dim, output_dim, embeddings):
    super(Net, self).__init__()
    self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)
    self.biLstm = nn.LSTM(embedding_size, hidden_dim, bidirectional=True, num_layers = 1, batch_first=True)
    self.fc = nn.Linear(hidden_dim*2, 128)
    self.dropout = nn.Dropout(0.33)
    self.classifier = nn.Linear(128, 9)
  def forward(self, sentence):
    embedding = self.embedding(sentence)
    lstm_out, (h_o, c_o) = self.biLstm(embedding)  #.view(len(sentence), 1, -1))
    lstm_out = self.dropout(lstm_out)
    dense_output = self.fc(lstm_out) #.view(len(sentence), -1))
    dense_output = self.dropout(dense_output)
    activations = F.elu(dense_output)
    final_outputs = self.classifier(activations)
    return final_outputs


# In[12]:


import torch.optim as optim

num_of_epoch = 30
embedding_size = 100
hidden_dim = 256
rho=0.05

embedding_matrix_tensor = torch.FloatTensor(embedding_matrix)
'''
model = Net(vocab_size, embedding_size, hidden_dim, tag_size, embedding_matrix_tensor).to(device)
loss_function = nn.CrossEntropyLoss()
optimiser = optim.SGD(model.parameters(), lr=0.3, momentum=0.85)
scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=1 - rho)


for epoch in range(num_of_epoch):
  print(epoch)
  model.train()
  total_loss=0
  for sentence, tags, lengths in train_loader:
    model.zero_grad()
    sentence = sentence.to(device)
    tags = tags.to(device)
    tag_scores = model(sentence)
    tag_scores = tag_scores.view(-1, tag_scores.shape[-1])
    tags = tags.view(-1)
    loss = loss_function(tag_scores, tags)
    loss.backward()
    optimiser.step()
    total_loss += loss.item()
  print(f"Epoch {epoch+1}/{num_of_epoch}, Loss: {total_loss/len(train_loader)}")
  scheduler.step()

'''
# In[13]:
model = Net(vocab_size, embedding_size, hidden_dim, tag_size, embedding_matrix_tensor)
model.load_state_dict(torch.load('blstm2.pt'))

dev_path = r'data/dev'
dev_data_df, w, t = process_input_from_file(dev_path)
dev_data_df['words_numerical'] = dev_data_df['sentence'].map(lambda seq: prepare_sequence(seq, word_to_idx))
dev_data_df["tags_numerical"] = dev_data_df['tags'].map(lambda tags: prepare_sequence(tags, tag_to_idx))

dev_dataset = NERDataset(dev_data_df["words_numerical"], dev_data_df["tags_numerical"])
dev_loader = DataLoader(dev_dataset, batch_size=8, collate_fn=pad_collate)


# In[14]:


idx_to_vocab = {idx: word for word, idx in word_to_idx.items()}
idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}

def write_predictions_to_file(model, data_loader, idx_to_tag, output_file_path, original_dev_sentences):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch_idx, (sentence_tensors, tags_tensors, lengths) in enumerate(data_loader):
#             sentence_tensors = sentence_tensors.to(device)
#             char_sentence_tensors = char_sentence_tensors.to(device)
            # sentence_tensors = sentence_tensors.to(device)
            # tags_tensors = tags_tensors.to(device)
            outputs = model(sentence_tensors)
            
            predicted_tag_indices = torch.argmax(outputs, dim=2)

            for i, length in enumerate(lengths):
                original_sentence = original_dev_sentences[batch_idx * data_loader.batch_size + i].split()
                for j in range(length):
                    original_word = original_sentence[j]
                    predicted_tag_index = predicted_tag_indices[i][j].item()
                    predicted_tag = idx_to_tag[predicted_tag_index]
                    predictions.append(f"{j+1} {original_word} {predicted_tag}\n")
                predictions.append("\n")

    with open(output_file_path, 'w') as writer:
        writer.writelines(predictions)

    print(f"Predictions written to {output_file_path}")


# In[16]:


# Define the path to the output file for predictions
output_file_path_glove = 'dev2.out'

# Write predictions to the file
write_predictions_to_file(model, dev_loader, idx_to_tag, output_file_path_glove, dev_data_df['sentence'])

# Define the paths to the predicted file and gold-standard file
predicted_file_path_glove = output_file_path_glove
gold_standard_file_path = 'data/dev'

# Run the eval.py script with the specified files
# get_ipython().system('python eval.py -p {predicted_file_path_glove} -g {gold_standard_file_path}')



# In[17]:


# torch.save(model.state_dict(), 'blstm2.pt')


# In[ ]:

def process_test_from_file(file_path):
    sentences = []
    tags = []
    with open(file_path, 'r') as file:
        data_blocks = file.read().strip().split('\n\n')
        
        for block in data_blocks:
            words = []
            word_tags = []
            for line in block.split('\n'):
                _, word = line.split()
                words.append(word)
                word_tags.append('O')
            sentences.append(' '.join(words))
            tags.append(' '.join(word_tags))
    
    return pd.DataFrame({
        'sentence': sentences,
        'tags': tags
    }), list(unique_words), list(unique_tags)

test_path = r'data/test'
test_data_df, w, t = process_test_from_file(test_path)
test_data_df['words_numerical'] = test_data_df['sentence'].map(lambda seq: prepare_sequence(seq, word_to_idx))
test_data_df["tags_numerical"] = test_data_df['tags'].map(lambda tags: prepare_sequence(tags, tag_to_idx))

test_dataset = NERDataset(test_data_df["words_numerical"], test_data_df["tags_numerical"])
test_loader = DataLoader(test_dataset, batch_size=8, collate_fn=pad_collate)

test_output = 'test2.out'
write_predictions_to_file(model, test_loader, idx_to_tag, test_output, test_data_df['sentence'])







