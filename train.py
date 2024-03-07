#!/usr/bin/env python
# coding: utf-8

# In[6]:


from model_bert import  BertModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader


model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


def tokenize_batch(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

# dataset = dataset.map(tokenize_batch, batched=True)

train_dataset = dataset['train']
test_dataset = dataset['test']

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

optimizer = AdamW(model.parameters(), lr=5e-5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

best_accuracy = 0.0

for epoch in range(3):
    model.train()
    for batch in train_loader:
        inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
        labels = batch['labels'].to(device)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Epoch {epoch + 1}, Validation Accuracy: {accuracy}')

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'best_bert_imdb_sentiment_model.pth')


# In[ ]:




