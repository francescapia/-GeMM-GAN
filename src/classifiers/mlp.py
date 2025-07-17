import torch
import torch.nn as nn
from typing import List
import numpy as np
from tqdm import tqdm
import random

class TorchMLPClassifier:
    def __init__(self, hidden_dims: List[int] = [], dropout_rate: float = 0.1, 
                 bn_momentum: float = 0.1, use_dropout: bool = True, use_norm: bool = True, learning_rate: float = 0.001,
                 num_epochs: int = 100, batch_size: int = 32, weight_decay: float = 1e-3, num_workers: int = 4, 
                 gradient_clipping: float = 10, device = 'cuda:0', verbose: bool = False, random_state: int = 42):
        super(TorchMLPClassifier, self).__init__()
        self.bn_momentum = bn_momentum
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dropout_rate = dropout_rate
        self.hidden_dims = hidden_dims
        self.use_dropout = use_dropout
        self.use_norm = use_norm
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.device = device
        self.gradient_clipping = gradient_clipping
        self.verbose = verbose
        self.random_state = random_state
        
    def build_model(self):
        layers = []            
        in_features = self.input_features
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            if self.use_norm:
                layers.append(nn.BatchNorm1d(hidden_dim, momentum=self.bn_momentum))
            layers.append(nn.ReLU())
            if self.use_dropout:
                layers.append(nn.Dropout(self.dropout_rate))
            in_features = hidden_dim
            
        layers.append(nn.Linear(in_features, self.num_classes))
        layers.append(nn.Softmax(dim=-1))
        self.model = nn.Sequential(*layers)
    
    def fit(self, x: np.ndarray, y: np.ndarray):
        self.input_features = x.shape[1]
        self.num_classes = y.max() + 1
        
        torch.manual_seed(self.random_state)
        random.seed(42)
        np.random.seed(42)
        
        self.build_model()
        self.model.train()
        self.model.to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        dataset = torch.utils.data.TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        
        iterator = tqdm(range(self.num_epochs), desc="Training", unit="epoch") if self.verbose else range(self.num_epochs)
        for epoch in iterator:
            for x_batch, y_batch in dataloader:
                optimizer.zero_grad()
                y_out = self.model(x_batch.to(self.device))
                loss = criterion(y_out, y_batch.to(self.device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                optimizer.step()
        return self
    
    def predict(self, x: np.ndarray, batch_size: int = 512):
        if x.shape[0] < batch_size:
            batch_size = x.shape[0]
        self.model.eval()
        dataset = torch.utils.data.TensorDataset(torch.tensor(x, dtype=torch.float32))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers)
        
        preds = []
        with torch.no_grad():
            for x_batch in dataloader:
                y_out = self.model(x_batch[0].to(self.device))
                preds.append(y_out.cpu().numpy().argmax(axis=1))
        return np.concatenate(preds, axis=0)
    
    def predict_proba(self, x: np.ndarray, batch_size: int = 512):
        if x.shape[0] < batch_size:
            batch_size = x.shape[0]
        self.model.eval()
        dataset = torch.utils.data.TensorDataset(torch.tensor(x, dtype=torch.float32))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=self.num_workers)
        
        preds = []
        with torch.no_grad():
            for x_batch in dataloader:
                y_out = self.model(x_batch[0].to(self.device))
                preds.append(y_out.cpu().numpy())
        return np.concatenate(preds, axis=0)
    
    
if __name__ == "__main__":
    fake_data = np.random.rand(2000, 18000)
    fake_labels = np.random.randint(0, 2, size=(2000,))
    model = TorchMLPClassifier(hidden_dims=[64, 32], dropout_rate=0.1, 
                               bn_momentum=0.1, use_dropout=True, use_norm=True, learning_rate=0.001,
                               num_epochs=50, batch_size=32, weight_decay=1e-3, num_workers=4, 
                               gradient_clipping=10, device='cuda:0', verbose=True)
    model.fit(fake_data, fake_labels)
    preds = model.predict(fake_data[:16])
    print(preds)
            