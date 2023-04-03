import numpy as np
import pandas as pd
import os
from scipy.stats import zscore 
from sklearn.cluster import KMeans 
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.pipeline import Pipeline 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, SequentialSampler 
from tqdm.auto import tqdm
from sklearn.model_selection import KFold
import random 
from transformers import get_linear_schedule_with_warmup 

def seed_everything(seed): 
    random.seed(seed) 
    os.environ["PYTHONHASHSEED"] = str(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = True 
    
seed_everything(42) 
train = pd.read_csv("train_data.csv") 
test = pd.read_csv("test_data.csv") 

class AutoEncoder(nn.Module): 
    def __init__(self, input_dim): 
        self.input_dim = input_dim
        super(AutoEncoder, self).__init__() 
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 6), 
            nn.ELU(), 
            nn.Linear(6, 4), 
            nn.ELU(), 
            nn.Linear(4,2), 
            nn.ELU()
        ) 
        self.decoder = nn.Sequential(
            nn.Linear(2, 4), 
            nn.ELU(), 
            nn.Linear(4, 6), 
            nn.ELU(), 
            nn.Linear(6, self.input_dim)
        ) 
    def forward(self, x):
        encoded = self.encoder(x) 
        decoded = self.decoder(encoded) 
        return decoded
    
kf = KFold(n_splits=10, shuffle=True, random_state=7789) 

device = torch.device("cuda") 

for idx, (train_idx, val_idx) in enumerate(kf.split(train)): 
    train_df, val_df = train.iloc[train_idx], train.iloc[val_idx] 
    pipeline = Pipeline([("normalizer", Normalizer()), ("scaler", MinMaxScaler())]) 
    pipeline.fit(train_df) 
    X_train_transformed = pipeline.transform(train_df) 
    X_val_transformed = pipeline.transform(val_df) 
    
    X_train_transformed = torch.tensor(X_train_transformed).float() 
    train_data = TensorDataset(X_train_transformed) 
    train_sampler = RandomSampler(train_data) 
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=32) 
    
    X_val_transformed = torch.tensor(X_val_transformed).float() 
    val_data = TensorDataset(X_val_transformed) 
    val_sampler = SequentialSampler(val_data) 
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=32) 
    
    features = X_train_transformed.shape[1] 
    model = AutoEncoder(features) 
    model.to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) 
    criterion = nn.MSELoss() 
    
    best_val_loss = 9999999999
    epochs = 2001 
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps = total_steps)  
    model.zero_grad() 
    for epoch in tqdm(range(1, epochs), position=0, leave=True, desc="Epochs"):
        train_loss = 0 
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch) 
            b_inputs = batch[0] 
            decoded = model(b_inputs)
            loss = criterion(decoded, b_inputs) 
            train_loss += loss.item() 
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
            optimizer.step() 
            scheduler.step() 
            model.zero_grad() 
        model.eval() 
        val_loss = 0 
        for step, batch in enumerate(val_dataloader): 
            batch = tuple(t.to(device) for t in batch) 
            b_inputs = batch[0] 
            with torch.no_grad(): 
                decoded = model(b_inputs) 
                loss = criterion(decoded, b_inputs) 
                val_loss += loss.item() 
        if val_loss < best_val_loss: 
            best_val_loss = val_loss 
            torch.save(model.state_dict(), f"KFOLD{idx+1}_autoencoder.pt") 
        if epoch > 0 and epoch % 100 == 0: 
            print(f"epoch : {epoch} | train loss : {train_loss} | best val loss : {best_val_loss}") 
            
    os.rename(f"KFOLD{idx+1}_autoencoder.pt", f"KFOLD{idx+1}_{best_val_loss}_autoencoder.pt")
    print("done!") 
        
