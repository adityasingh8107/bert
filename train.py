#!/usr/bin/env python
# coding: utf-8

# In[6]:


from model_bert import  BertModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader


dataset = load_dataset('imdb')
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


def tokenize_batch(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

dataset = dataset.map(tokenize_batch, batched=True)

train_dataset = dataset['train']
test_dataset = dataset['test']

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

optimizer = AdamW(model.parameters(), lr=5e-5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

best_accuracy = 0.0

model.train()
for epoch in range(3):  # You can adjust the number of epochs
    total_loss = 0.0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()
        total_loss += loss.item()

        # Update parameters
        optimizer.step()
        optimizer.zero_grad()

        if batch%10000 == 0:
            print("Loss", total_loss)

    # Print average loss for each epoch
    avg_train_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}, Average Training Loss: {avg_train_loss}')
    
    accuracy = correct / total
    print(f'Epoch {epoch + 1}, Validation Accuracy: {accuracy}')

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'best_bert_imdb_sentiment_model.pth')


# In[ ]:




