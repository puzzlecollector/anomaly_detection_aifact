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

best_val_losses = [0.00045378284994512796,
                   0.0073279617936350405,
                   0.0025535928543831687,
                   0.0010728383185778512,
                   0.003940425798646174,
                   0.001941366477694828,
                   0.009518826111161616,
                   0.0028283100946282502,
                   0.0034261311921000015,
                   0.000487238954519853] 

criterion = nn.MSELoss() 
device = torch.device("cuda")

all_test_mse_scores = np.zeros((test.shape[0]))

for idx, (train_idx, val_idx) in enumerate(kf.split(train)): 
    train_df, val_df = train.iloc[train_idx], train.iloc[val_idx] 
    pipeline = Pipeline([("normalizer", Normalizer()), ("scaler", MinMaxScaler())]) 
    pipeline.fit(train_df) 
    X_train_transformed = pipeline.transform(train_df) 
    X_val_transformed = pipeline.transform(val_df) 
    
    features = X_train_transformed.shape[1] 
    checkpoint = torch.load(f"KFOLD{idx+1}_{best_val_losses[idx]}_autoencoder.pt")
    model = AutoEncoder(features) 
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval() 
       
    X_test_transformed = pipeline.transform(test) 
    X_test_transformed = torch.tensor(X_test_transformed).float() 
    test_data = TensorDataset(X_test_transformed) 
    test_sampler = SequentialSampler(test_data) 
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=1) 
    
    test_mse_scores = [] 
    for step, batch in tqdm(enumerate(test_dataloader), position=0, desc=f"testing model fold {idx+1}", total=len(test_dataloader)):
        batch = tuple(t.to(device) for t in batch) 
        b_inputs = batch[0] 
        with torch.no_grad(): 
            decoded = model(b_inputs) 
            loss = criterion(decoded, b_inputs)
            test_mse_scores.append(loss.item())  
    
    for i in range(len(test_mse_scores)): 
        all_test_mse_scores[i] += test_mse_scores[i] 
        
for i in range(len(all_test_mse_scores)): 
    all_test_mse_scores[i] /= 10.0 

# make prediciton 
gamma = 3 
def mad_score(points): 
    m = np.median(points)
    ad = np.abs(points - m) 
    mad = np.median(ad) 
    return 0.6745 * ad / mad 
z_scores = mad_score(all_test_mse_scores)    

outliers = z_scores > gamma 
outliers = outliers.astype(int)

submission = pd.read_csv("answer_sample.csv") 
submission["label"] = outliers 

submission.to_csv("autoencoder_ensemble.csv", index=False) 
print("done!") 
