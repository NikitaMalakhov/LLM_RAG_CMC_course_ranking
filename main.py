import streamlit as st

from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List
import pickle

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-large').to(device)


with open("./notebooks/embeddings_db.pkl", "rb") as f:
    database = pickle.load(f)


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def rank(sentence: str, k: int=5) -> List[str]:
    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    sentence = f'query: {sentence}'
    tokenized_sentences = tokenizer(
        [sentence], max_length=512,
        padding=True, truncation=True,
        return_tensors='pt'
    )
    tokenized_sentences['input_ids'] = tokenized_sentences['input_ids'].to(device)
    tokenized_sentences['attention_mask'] = tokenized_sentences['attention_mask'].to(device)
    out = model(**tokenized_sentences)
    embeddings = average_pool(out.last_hidden_state, tokenized_sentences['attention_mask'])
    embeddings = embeddings.detach().cpu()
    search_metrics = {}

    for sentence in database.keys():
        metric = F.cosine_similarity(embeddings, database[sentence])
        search_metrics[sentence] = metric.item()

    sorted_scores = sorted(search_metrics.items(), key=lambda x: x[1], reverse=True)
    return [(sentence, score) for sentence, score in sorted_scores[:k]]

st.title('Ранжировочная система')

input_string = st.text_input('Запрос:')

if st.button('Найти'):
    if input_string:
        sentences = rank(input_string, k=5)

        st.write('Output: ', [k[0] for k in sentences])