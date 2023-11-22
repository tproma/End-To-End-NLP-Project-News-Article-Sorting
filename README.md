# End-To-End-NLP-Project-News-Article-Sorting

### Step 1: Create condsa environment
```
conda create -n textSort python=3.8 -y
```
```
conda activate textSort
```

### Step 2: Install the requirements
```
pip install -r requirements.txt
```

# For pytorch cuda version
``` 
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

```
pip install -U scikit-learn
```

#Necessary for running the training arguments
```
pip install --upgrade accelerate
pip install -y transformers accelerate
pip install transformers accelerate
```

## Workflows
- update config.yaml
- update  params.yaml
- update entity
- update the configuration manager in src config
- update the components
- update the pipelines
- update main.py
- update app.py


### ARTICLES
    - How to iterate through the dataset using DataLoader
    https://towardsdatascience.com/how-to-use-datasets-and-dataloader-in-pytorch-for-custom-text-data-270eed7f7c00