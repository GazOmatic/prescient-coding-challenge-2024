import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset

### Imports
from collections import deque
from typing import Dict, Optional, Literal
import torch
import torch.nn as nn


### Grokfast
def gradfilter_ema(
    m: nn.Module,
    grads: Optional[Dict[str, torch.Tensor]] = None,
    alpha: float = 0.99,
    lamb: float = 5.0,
) -> Dict[str, torch.Tensor]:
    if grads is None:
        grads = {n: p.grad.data.detach() for n, p in m.named_parameters() if p.requires_grad}

    for n, p in m.named_parameters():
        if p.requires_grad:
            grads[n] = grads[n] * alpha + p.grad.data.detach() * (1 - alpha)
            p.grad.data = p.grad.data + grads[n] * lamb

    return grads


### Grokfast-MA
def gradfilter_ma(
    m: nn.Module,
    grads: Optional[Dict[str, deque]] = None,
    window_size: int = 128,
    lamb: float = 5.0,
    filter_type: Literal['mean', 'sum'] = 'mean',
    warmup: bool = True,
    trigger: bool = False,
) -> Dict[str, deque]:
    if grads is None:
        grads = {n: deque(maxlen=window_size) for n, p in m.named_parameters() if p.requires_grad}

    for n, p in m.named_parameters():
        if p.requires_grad:
            grads[n].append(p.grad.data.detach())

            if not warmup or len(grads[n]) == window_size and not trigger:
                if filter_type == "mean":
                    avg = sum(grads[n]) / len(grads[n])
                elif filter_type == "sum":
                    avg = sum(grads[n])
                else:
                    raise ValueError(f"Unrecognized filter_type {filter_type}")
                p.grad.data = p.grad.data + avg * lamb

    return grads

# Define the AttentionalMLP model
class AttentionalMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(AttentionalMLP, self).__init__()
        # MLP Layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # Single-head Attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1, batch_first=True)
        
        # Projection Layer
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Parameters:
        - x: Tensor of shape (batch_size, input_dim)
        
        Returns:
        - logits: Tensor of shape (batch_size, num_classes)
        """
        # MLP Layer with ReLU activation
        hidden = F.relu(self.fc1(x))  # Shape: (batch_size, hidden_dim)
        
        # Add sequence dimension for attention: (batch_size, 1, hidden_dim)
        hidden = hidden.unsqueeze(1)
        
        # Apply attention (self-attention)
        attn_output, _ = self.attention(hidden, hidden, hidden)  # Shape: (batch_size, 1, hidden_dim)
        
        # Remove sequence dimension
        attn_output = attn_output.squeeze(1)  # Shape: (batch_size, hidden_dim)
        
        # Projection Layer
        logits = self.fc2(attn_output)  # Shape: (batch_size, num_classes)
        
        return logits

# Define the Scikit-Learn compatible classifier
class AttentionalMLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, hidden_dim=32, num_classes=2, 
                 lr=0.001, batch_size=32, epochs=100, 
                 verbose=False, device=None, random_state=None):
        """
        Initialize the classifier.
        
        Parameters:
        - input_dim: Number of input features.
        - hidden_dim: Number of hidden units in the MLP layer.
        - num_classes: Number of target classes.
        - lr: Learning rate.
        - batch_size: Batch size for training.
        - epochs: Number of training epochs.
        - verbose: If True, print training progress.
        - device: 'cpu' or 'cuda'. If None, automatically detected.
        - random_state: Seed for reproducibility.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_state = random_state
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        
        # Initialize the model
        self.model_ = AttentionalMLP(input_dim, hidden_dim, num_classes).to(self.device)
        
        # Define loss and optimizer
        self.criterion_ = nn.CrossEntropyLoss()
        self.optimizer_ = torch.optim.SGD(self.model_.parameters(), lr=self.lr)
        
    def fit(self, X, y):
        """
        Fit the model to the data.
        
        Parameters:
        - X: Array-like of shape (n_samples, n_features)
        - y: Array-like of shape (n_samples,)
        
        Returns:
        - self
        """
        # Convert X and Y to tensor
        X = np.array(X)
        y = np.array(y)


        if self.random_state:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y_encoded, dtype=torch.long)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model_.train()
        grads=None
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Zero gradients
                self.optimizer_.zero_grad()
                
                # Forward pass
                outputs = self.model_(batch_X)
                
                # Compute loss
                loss = self.criterion_(outputs, batch_y)
                
                # Backward pass and optimization
                loss.backward()
                grads = gradfilter_ema(self.model_, grads=grads, alpha=0.99, lamb=5)

                self.optimizer_.step()
                
                epoch_loss += loss.item() * batch_X.size(0)
            
            epoch_loss /= len(dataloader.dataset)
            
            if self.verbose and (epoch + 1) % 1 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters:
        - X: Array-like of shape (n_samples, n_features)
        
        Returns:
        - proba: Array-like of shape (n_samples, n_classes)
        """
        # Convert X to tensor
        X = np.array(X)
        
        self.model_.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model_(X_tensor)
            proba = F.softmax(logits, dim=1)
            return proba.cpu().numpy()
    
    def predict(self, X):
        """
        Predict class labels for X.
        
        Parameters:
        - X: Array-like of shape (n_samples, n_features)
        
        Returns:
        - predictions: Array-like of shape (n_samples,)
        """
        proba = self.predict_proba(X)
        class_indices = np.argmax(proba, axis=1)
        return self.label_encoder.inverse_transform(class_indices)

# Example usage
if __name__ == "__main__":
    # Generate a synthetic classification dataset
    X, y = make_classification(n_samples=1000, n_features=20, 
                               n_informative=15, n_redundant=5, 
                               n_classes=3, random_state=42)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state=42)
    
    # Initialize the classifier
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    classifier = AttentionalMLPClassifier(input_dim=input_dim, 
                                         hidden_dim=64, 
                                         num_classes=num_classes, 
                                         lr=0.001, 
                                         batch_size=32, 
                                         epochs=100, 
                                         verbose=True, 
                                         random_state=42)
    
    # Fit the model
    classifier.fit(X_train, y_train)
    
    # Predict probabilities
    proba = classifier.predict_proba(X_test)
    print("Predicted probabilities:\n", proba[:5])
    
    # Predict class labels
    y_pred = classifier.predict(X_test)
    print("Predicted labels:\n", y_pred[:5])
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
