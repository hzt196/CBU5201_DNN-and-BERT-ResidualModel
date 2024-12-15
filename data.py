import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class NewsData(Dataset):
    def __init__(self, sentences, labels, tokenizer, model, max_length):
        self.labels = labels
        self.texts = []
        for sentence in sentences:
            text = sentence[:max_length].ljust(max_length)
            input_ids = torch.tensor(tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=True,
                padding='max_length',
                max_length=max_length
            )).unsqueeze(0)
            with torch.no_grad():
                outputs = model(input_ids)
            cls_embedding = outputs[0][:, 0, :].squeeze().numpy()
            self.texts.append(torch.tensor(cls_embedding))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], torch.tensor(self.labels[idx], dtype=torch.long)

def load_data(filepath, tokenizer, model, max_length=128):
    data = pd.read_excel(filepath)
    sentences = data['text'].tolist()
    labels = data['label'].tolist()
    return NewsData(sentences, labels, tokenizer, model, max_length)

def get_dataloaders(filepath, tokenizer, model, batch_size=32, max_length=128):
    dataset = load_data(filepath, tokenizer, model, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
