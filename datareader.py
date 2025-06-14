import random
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
import os
import json
from nltk.corpus import stopwords
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer
import argparse

# Download resource NLTK yang diperlukan
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')

# try:
#     nltk.data.find('corpora/stopwords')  
# except LookupError:
#     nltk.download('stopwords')

# try:
#     nltk.data.find('tokenizers/punkt_tab')
# except LookupError:
#     nltk.download('punkt_tab')

# try:
#     nltk.data.find('corpora/wordnet')
# except LookupError:
#     nltk.download('wordnet')

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

INDONESIAN_STOPWORDS = set(stopwords.words('indonesian'))


class YoutubeCommentDataset(Dataset):
    def __init__(
        self,
        file_path="dataset/dataset_judol.xlsx",
        # tokenizer_name="D:/TA/code/indobert-base-p1",
        tokenizer_name="indobenchmark/indobert-base-p1",
        folds_file="youtube_datareader-simple-folds.json",
        random_state=2003,
        split="train",
        fold=0,
        augmentasi_file="augmentasi.json",
        augment_prob=1.0,
    ):
        super(YoutubeCommentDataset, self).__init__()

        self.augment_prob = augment_prob
        self.file_path = file_path
        self.folds_file = folds_file
        self.random_state = random_state
        self.split = split
        self.fold = fold
        self.augmentasi_data = self.load_augmentasi(augmentasi_file)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.load_data()
        self.Setup_folds()
        self.setup_indices()

    def load_augmentasi(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def random_typo(self, text):
        words = text.split()
        if len(words) < 1:
            return text
        idx = random.randint(0, len(words) - 1)
        word = words[idx]
        if len(word) > 1:
            char_list = list(word)
            i = random.randint(0, len(char_list) - 2)
            char_list[i], char_list[i+1] = char_list[i+1], char_list[i]
            words[idx] = ''.join(char_list)
        return ' '.join(words)

    def random_swap(self, text):
        words = text.split()
        if len(words) < 2:
            return text
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)

    def random_delete(self, text):
        words = text.split()
        if len(words) <= 1:
            return text
        idx = random.randint(0, len(words) - 1)
        del words[idx]
        return ' '.join(words)

    def augment_text(self, text):
        for phrase, replacements in self.augmentasi_data.get("replace_phrases", {}).items():
            if phrase in text:
                text = text.replace(phrase, random.choice(replacements))
        words = text.split()
        for i, word in enumerate(words):
            if word in self.augmentasi_data.get("synonyms", {}):
                words[i] = random.choice(self.augmentasi_data["synonyms"][word])
        text = ' '.join(words)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in INDONESIAN_STOPWORDS]
        text = ' '.join(tokens)
        text = self.random_typo(text)
        text = self.random_swap(text)
        text = self.random_delete(text)
        return text

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx = self.indices[idx]
        komentar = str(self.df.iloc[idx]['comment'])
        kategori = self.df.iloc[idx]['kategori']

        comment_processed = self.preprocess_text(komentar)

        if random.random() < self.augment_prob:
            comment_processed = self.augment_text(comment_processed)

        encoding = self.tokenizer(
            comment_processed,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        kategori_mapping = {
            'judol': 0,
            'non-judol': 1,
        }

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(kategori_mapping.get(kategori, -1), dtype=torch.long),
            'original_text': komentar,
            'processed_text': comment_processed,
            'original_index': idx
        }

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        words = word_tokenize(text)
        for phrase, replacements in self.augmentasi_data["replace_phrases"].items():
            if phrase in text:
                text = text.replace(phrase, random.choice(replacements))
        words = [kata for kata in words if kata not in INDONESIAN_STOPWORDS]
        return ' '.join(words)

    def setup_indices(self):
        fold_key = f"fold_{self.fold}"
        if self.split == "train":
            self.indices = self.fold_indices[fold_key]['train_indices']
        else:
            self.indices = self.fold_indices[fold_key]['val_indices']

    def Setup_folds(self):
        if os.path.exists(self.folds_file):
            self.load_folds()
        else:
            self.create_folds()

    def load_folds(self):
        with open(self.folds_file, 'r') as f:
            fold_data = json.load(f)
        self.fold_indices = fold_data['fold_indices']

    def create_folds(self):
        print(f"membuat 5-folds CV dengan random state {self.random_state}")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        fold_indices = {}
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.df, self.df['kategori'])):
            fold_indices[f"fold_{fold}"] = {
                'train_indices': train_idx.tolist(),
                'val_indices': val_idx.tolist()
            }
        with open(self.folds_file, 'w') as f:
            json.dump({
                'fold_indices': fold_indices,
                'n_samples': len(self.df),
                'n_folds': 5,
                'random_state': self.random_state
            }, f)
        self.fold_indices = fold_indices

    def load_data(self):
        self.df = pd.read_excel(self.file_path)
        self.df.columns = ['id', 'userName', 'comment', 'kategori']
        self.df = self.df.dropna(subset=['comment', 'kategori'])
        self.df['kategori'] = self.df['kategori'].astype(str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment_prob", type=float, default=1.0, help="Probability of applying augmentation")
    args = parser.parse_args()

    train_dataset = YoutubeCommentDataset(fold=0, split="train", augment_prob=args.augment_prob)
    val_dataset = YoutubeCommentDataset(fold=0, split="val", augment_prob=0.0)
    data = train_dataset[8]
    print(f"Input IDs: {data['input_ids']}")
    print(f"Original Text: {data['original_text']}")
    print(f"Processed Text: {data['processed_text']}")
    print(f"Original Index: {data['original_index']}")
    print(f"Labels: {data['labels']}")
