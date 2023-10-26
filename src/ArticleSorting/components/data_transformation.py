import os
from ArticleSorting.logging import logger

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset

import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        

    def encode_categories(self):
        self.df = pd.read_csv(self.config.data_path)
        self.df['encoded_label'] = self.df['Category'].astype('category').cat.codes
        hg_data = Dataset.from_pandas(self.df)
        print(hg_data)
        return hg_data
   
    def tokenize_dataset(self, data):
        return self.tokenizer(data["Text"],
                     max_length=512,
                     truncation=True,
                     padding="max_length")
    
    def convert(self):
        data = self.encode_categories()
        tokenized_data = data.map(self.tokenize_dataset)
        # Remove  columnms
        tokenized_data = tokenized_data.remove_columns(["ArticleId", "Text", "Category"])
        # Rename label to labels because the model expects the name labels
        dataset = tokenized_data.rename_column("encoded_label", "labels")
        # Change the format to PyTorch tensors
        dataset.set_format("torch")
        print(dataset)

        dataset.save_to_disk(os.path.join(self.config.root_dir,"BBC dataset"))
