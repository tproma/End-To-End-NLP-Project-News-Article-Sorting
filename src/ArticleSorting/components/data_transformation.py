import os
from ArticleSorting.logging import logger
from ArticleSorting.entity import DataTransformationConfig

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
        
        df = pd.read_csv(self.config.data_path)
        df['encoded_label'] = df['Category'].astype('category').cat.codes

        ## Spliting the Data
        # Training dataset
        train_data = df.sample(frac=0.8, random_state=42)
        # Testing dataset
        test_data = df.drop(train_data.index)

        # Convert pyhton dataframe to Hugging Face arrow dataset
        hg_train_data = Dataset.from_pandas(train_data)
        hg_test_data = Dataset.from_pandas(test_data)
        print(hg_train_data, hg_test_data)
        return hg_train_data, hg_test_data



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
