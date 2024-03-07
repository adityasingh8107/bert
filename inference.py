#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from transformers import BertTokenizer
from model_bert import  BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

model.load_state_dict(torch.load('best_bert_imdb_sentiment_model.pth'))
model.eval()

def predict_sentiment(review):
    inputs = tokenizer(review, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
    return {'positive': probabilities[0][1], 'negative': probabilities[0][0]}

review = "I really enjoyed this movie. The acting was superb."
sentiment = predict_sentiment(review)
print(sentiment)

