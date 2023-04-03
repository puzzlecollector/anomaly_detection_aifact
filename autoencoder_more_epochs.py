import numpy as np
import pandas as pd
from scipy.stats import zscore 
from sklearn.cluster import KMeans 
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.pipeline import Pipeline 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, SequentialSampler 
from tqdm.auto import tqdm

train = pd.read_csv("train_data.csv") 
test = pd.read_csv("test_data.csv") 

pipeline = Pipeline([("normalizer", Normalizer()), 
                     ("scaler", MinMaxScaler())]) 

pipeline.fit(train)

X_train_transformed = pipeline.transform(train) 

class AutoEncoder(nn.Module): 
    def __init__(self, input_dim): 
        self.input_dim = input_dim
        super(AutoEncoder, self).__init__() 
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 6), 
            nn.ELU(), 
            nn.Linear(6, 4), 
            nn.ELU(), 
            nn.Linear(4, 2), 
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
    
model = AutoEncoder(X_train_transformed.shape[1]) 
X_train_transformed = torch.tensor(X_train_transformed).float() 
train_data = TensorDataset(X_train_transformed) 
train_sampler = RandomSampler(train_data) 
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=32) 

device = torch.device("cuda") 
model = model.to(device) 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss() 

for epoch in tqdm(range(1, 10001), position=0, leave=True, desc="Epochs"):
    train_loss = 0 
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch) 
        b_inputs = batch[0] 
        decoded = model(b_inputs)
        loss = criterion(decoded, b_inputs) 
        train_loss += loss.item() 
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step()  
    if epoch > 0 and epoch % 100 == 0: 
        print(f"epoch : {epoch} | train loss : {train_loss}") 
        
torch.save(model.state_dict(), "autoencoder_more_epochs_chkpt.pt")

X_test_transformed = pipeline.transform(test) 
X_test_transformed = torch.tensor(X_test_transformed).float() 

test_data = TensorDataset(X_test_transformed) 
test_sampler = SequentialSampler(test_data) 
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=1) 

test_mse_scores = [] 

model.eval() 

with torch.no_grad(): 
    for step, batch in tqdm(enumerate(test_dataloader), position=0, leave=True, desc="Reconstructing test dataset"): 
        batch = tuple(t.to(device) for t in batch) 
        b_inputs = batch[0] 
        decoded = model(b_inputs) 
        loss = criterion(decoded, b_inputs)
        test_mse_scores.append(loss.item()) 
        
# using the median absolute deviation method to define outliers 

gamma = 3 

def mad_score(points): 
    m = np.median(points)
    ad = np.abs(points - m) 
    mad = np.median(ad) 
    return 0.6745 * ad / mad 


z_scores = mad_score(test_mse_scores) 

outliers = z_scores > gamma 

outliers = outliers.astype(int) 

# 정상: 0, 이상: 1 
submission = pd.read_csv("answer_sample.csv") 
submission["label"] = outliers 
submission.to_csv("autoencoders_10000_epochs.csv", index=False) 

