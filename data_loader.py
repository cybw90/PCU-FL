
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

class EdgeIIoTDataset(Dataset):
    """Edge-IIoT dataset with IID/non-IID distribution options"""
    
    def __init__(self, data_path="LP.csv", client_id=None, train=True, 
                 normalize=True, num_clients=100, task_mode="binary_dominant",
                 data_distribution="iid"):
        """
        Args:
            data_path: Path to *.csv file
            client_id: Client ID for data partitioning
            train: Whether to load train or test data
            normalize: Whether to normalize features
            num_clients: Total number of clients
            task_mode: "multi_class", "binary_dominant", "binary_anomaly", or "superclass"
            data_distribution: "iid", "non-iid-balanced", "non-iid", or "extreme-non-iid"
        """
        self.client_id = client_id
        self.train = train
        self.normalize = normalize
        self.num_clients = num_clients
        self.task_mode = task_mode
        self.data_distribution = data_distribution
        
        # Load and preprocess data
        self.data, self.labels, self.scaler, self.num_classes = self._load_and_preprocess(
            data_path, client_id, train
        )
        
    def _load_and_preprocess(self, data_path, client_id, train):
        """Load LP.csv and preprocess for federated learning"""
        
        # Check if file exists
        if not os.path.exists(data_path):
            if os.path.exists("*.csv"):
                data_path = "*.csv"
            else:
                raise FileNotFoundError(f"*.csv not found")
            
        print(f"Loading Edge-IIoT data from {data_path}")
        
        # Load data
        df = pd.read_csv(data_path, low_memory=False)
        
        # If dataset is large, sample for test set
        if len(df) > 50000 and not train:
            df = df.sample(n=min(10000, len(df)), random_state=42)
        
        print(f"Dataset shape: {df.shape}")
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Identify feature columns and label column
        feature_cols = df.columns[:-1].tolist()
        label_col = df.columns[-1]
        
        print(f"Number of features: {len(feature_cols)}")
        print(f"Label column: {label_col}")
        
        # Drop rows with missing values
        initial_size = len(df)
        df = df.dropna()
        if len(df) < initial_size:
            print(f"Dropped {initial_size - len(df)} rows with missing values")
        
        # Handle categorical features
        for col in feature_cols:
            if df[col].dtype == 'object':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
        
        # Extract features
        X = df[feature_cols].values.astype(np.float32)
        
        # Handle labels based on task mode
        y = df[label_col].values
        if y.dtype == 'object' or y.dtype.name == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
            original_num_classes = len(le.classes_)
            print(f"Original classes found: {le.classes_[:10]}...")
        else:
            y = y.astype(np.int64)
            original_num_classes = len(np.unique(y))
        
        # Transform labels based on task mode
        y, num_classes = self._transform_labels(y, original_num_classes)
        
        print(f"Task mode: {self.task_mode}")
        print(f"Number of classes after transformation: {num_classes}")
        unique, counts = np.unique(y, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  Class {u}: {c} samples ({c/len(y)*100:.1f}%)")
        
        # BALANCE THE DATASET FOR BINARY CLASSIFICATION
        if num_classes == 2 and self.task_mode == "binary_dominant":
            class_0_idx = np.where(y == 0)[0]
            class_1_idx = np.where(y == 1)[0]
            
            # Undersample majority class to match minority
            n_minority = len(class_1_idx)
            n_majority = len(class_0_idx)
            
            if n_majority > n_minority:
                # Undersample class 0 to match class 1 size
                class_0_sampled = np.random.choice(class_0_idx, n_minority, replace=False)
                balanced_idx = np.concatenate([class_0_sampled, class_1_idx])
            else:
                # Undersample class 1 to match class 0 size (rare case)
                class_1_sampled = np.random.choice(class_1_idx, n_majority, replace=False)
                balanced_idx = np.concatenate([class_0_idx, class_1_sampled])
            
            np.random.shuffle(balanced_idx)
            
            X = X[balanced_idx]
            y = y[balanced_idx]
            
            print(f"Balanced dataset: {len(X)} samples")
            
            # Print new distribution
            unique, counts = np.unique(y, return_counts=True)
            for u, c in zip(unique, counts):
                print(f"  After balancing - Class {u}: {c} samples ({c/len(y)*100:.1f}%)")
        
        # Normalize features
        scaler = None
        if self.normalize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
            # Check for NaN after normalization
            if np.isnan(X).any():
                print("Warning: NaN values after normalization, replacing with 0")
                X = np.nan_to_num(X, nan=0.0)
        
        # Partition data for federated learning
        if client_id is not None:
            X, y = self._partition_data(X, y, client_id, train, num_classes)
        else:
            # For server test set
            if not train:
                if len(X) > 5000:
                    try:
                        X, _, y, _ = train_test_split(
                            X, y, test_size=max(0.5, 1 - 5000/len(X)), 
                            stratify=y, random_state=42
                        )
                    except:
                        indices = np.random.choice(len(X), min(5000, len(X)), replace=False)
                        X = X[indices]
                        y = y[indices]
        
        return X, y, scaler, num_classes
    
    def _transform_labels(self, y, original_num_classes):
        """Transform labels based on task mode"""
        
        if self.task_mode == "multi_class":
            return y, original_num_classes
            
        elif self.task_mode == "binary_dominant":
            unique, counts = np.unique(y, return_counts=True)
            dominant_class = unique[np.argmax(counts)]
            y_binary = (y == dominant_class).astype(np.int64)
            print(f"Binary classification: Class {dominant_class} (positive) vs rest (negative)")
            return y_binary, 2
            
        elif self.task_mode == "binary_anomaly":
            unique, counts = np.unique(y, return_counts=True)
            normal_classes = unique[counts > np.percentile(counts, 50)]
            y_binary = np.isin(y, normal_classes).astype(np.int64)
            print(f"Anomaly detection: {len(normal_classes)} normal classes vs anomalies")
            return y_binary, 2
            
        elif self.task_mode == "superclass":
            unique, counts = np.unique(y, return_counts=True)
            sorted_indices = np.argsort(counts)[::-1]
            sorted_classes = unique[sorted_indices]
            sorted_counts = counts[sorted_indices]
            
            total = len(y)
            superclass_map = {}
            cumsum = 0
            superclass = 0
            
            for cls, cnt in zip(sorted_classes, sorted_counts):
                superclass_map[cls] = superclass
                cumsum += cnt
                if cumsum > total * 0.33 and superclass == 0:
                    superclass = 1
                elif cumsum > total * 0.66 and superclass == 1:
                    superclass = 2
            
            y_super = np.array([superclass_map[label] for label in y])
            num_superclasses = len(np.unique(y_super))
            print(f"Grouped {original_num_classes} classes into {num_superclasses} superclasses")
            return y_super, num_superclasses
        else:
            raise ValueError(f"Unknown task_mode: {self.task_mode}")
    
    def _partition_data(self, X, y, client_id, train, num_classes):
        """Partition data with IID/non-IID distribution options"""
        np.random.seed(42 + client_id)
        n_samples = len(X)
        
        print(f"Data distribution mode: {self.data_distribution}")
        
        if self.data_distribution == "iid":
            # IID: Each client gets random samples with same distribution as global
            samples_per_client = max(100, n_samples // self.num_clients)
            all_indices = np.arange(n_samples)
            client_indices = np.random.choice(all_indices, samples_per_client, replace=True)
            
        elif self.data_distribution == "non-iid-balanced":
            # Non-IID but balanced: Each client gets 50/50 split
            if num_classes == 2:
                class_0_indices = np.where(y == 0)[0]
                class_1_indices = np.where(y == 1)[0]
                
                samples_per_client = max(100, n_samples // self.num_clients)
                n_per_class = samples_per_client // 2
                
                selected_0 = np.random.choice(class_0_indices, n_per_class, replace=True)
                selected_1 = np.random.choice(class_1_indices, n_per_class, replace=True)
                client_indices = np.concatenate([selected_0, selected_1])
            else:
                # Multi-class balanced
                samples_per_client = max(100, n_samples // self.num_clients)
                samples_per_class = samples_per_client // num_classes
                client_indices = []
                
                for c in range(num_classes):
                    class_c_indices = np.where(y == c)[0]
                    if len(class_c_indices) > 0:
                        selected = np.random.choice(class_c_indices, samples_per_class, replace=True)
                        client_indices.extend(selected)
                client_indices = np.array(client_indices)
                
        elif self.data_distribution == "non-iid":
            # Non-IID: Different distributions per client (current implementation)
            if num_classes == 2:
                class_0_indices = np.where(y == 0)[0]
                class_1_indices = np.where(y == 1)[0]
                
                # Beta distribution for variety
                alpha, beta = 2 + (client_id % 5), 2 + ((client_id + 2) % 5)
                class_0_ratio = np.random.beta(alpha, beta)
                class_0_ratio = np.clip(class_0_ratio, 0.2, 0.8)
                
                samples_per_client = max(100, n_samples // self.num_clients * 2)
                n_class_0 = int(samples_per_client * class_0_ratio)
                n_class_1 = samples_per_client - n_class_0
                
                selected_0 = np.random.choice(class_0_indices, n_class_0, replace=True)
                selected_1 = np.random.choice(class_1_indices, n_class_1, replace=True)
                client_indices = np.concatenate([selected_0, selected_1])
            else:
                # Multi-class non-IID with Dirichlet
                alpha = 0.5  # Lower alpha = more non-IID
                samples_per_client = max(100, n_samples // self.num_clients)
                
                # Sample from Dirichlet for class proportions
                proportions = np.random.dirichlet([alpha] * num_classes)
                client_indices = []
                
                for c in range(num_classes):
                    class_c_indices = np.where(y == c)[0]
                    n_samples_c = int(samples_per_client * proportions[c])
                    if len(class_c_indices) > 0 and n_samples_c > 0:
                        selected = np.random.choice(class_c_indices, n_samples_c, replace=True)
                        client_indices.extend(selected)
                client_indices = np.array(client_indices)
                
        elif self.data_distribution == "extreme-non-iid":
            # Extreme non-IID: Some clients only see one class
            if num_classes == 2:
                if client_id % 3 == 0:
                    # Only class 0
                    client_indices = np.where(y == 0)[0]
                elif client_id % 3 == 1:
                    # Only class 1
                    client_indices = np.where(y == 1)[0]
                else:
                    # Mixed 50/50
                    class_0 = np.where(y == 0)[0]
                    class_1 = np.where(y == 1)[0]
                    n_each = 100
                    selected_0 = np.random.choice(class_0, n_each, replace=True)
                    selected_1 = np.random.choice(class_1, n_each, replace=True)
                    client_indices = np.concatenate([selected_0, selected_1])
            else:
                # Each client gets 1-2 classes only
                client_class = client_id % num_classes
                client_indices = np.where(y == client_class)[0]
            
            # Ensure minimum samples
            n_samples = min(200, len(client_indices))
            client_indices = np.random.choice(client_indices, n_samples, replace=True)
        else:
            raise ValueError(f"Unknown data_distribution: {self.data_distribution}")
        
        # Shuffle client's data
        np.random.shuffle(client_indices)
        
        # Get client's subset
        client_X = X[client_indices]
        client_y = y[client_indices]
        
        # Log class distribution for this client
        if client_id == 0 or client_id == self.num_clients - 1:
            unique_classes, class_counts = np.unique(client_y, return_counts=True)
            print(f"  Client {client_id} class distribution:")
            for cls, cnt in zip(unique_classes, class_counts):
                print(f"    Class {cls}: {cnt} ({cnt/len(client_y)*100:.1f}%)")
        
        # Train/test split
        split_idx = int(0.8 * len(client_X))
        
        if train:
            return client_X[:split_idx], client_y[:split_idx]
        else:
            return client_X[split_idx:], client_y[split_idx:]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Return a single sample"""
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample, label
    
    def get_input_dim(self):
        """Get input dimension for model initialization"""
        return self.data.shape[1]
    
    def get_num_classes(self):
        """Get number of classes for model initialization"""
        return self.num_classes

def get_edge_iiot_loaders(config, client_id=None):
    """Get data loaders with task mode and distribution from config"""
    
    data_file = "LP.csv"
    task_mode = getattr(config, 'task_mode', 'binary_dominant')
    data_distribution = getattr(config, 'data_distribution', 'iid')
    
    # Create datasets
    train_dataset = EdgeIIoTDataset(
        data_path=data_file,
        client_id=client_id,
        train=True,
        normalize=True,
        num_clients=config.num_clients,
        task_mode=task_mode,
        data_distribution=data_distribution
    )
    
    test_dataset = EdgeIIoTDataset(
        data_path=data_file,
        client_id=client_id,
        train=False,
        normalize=True,
        num_clients=config.num_clients,
        task_mode=task_mode,
        data_distribution=data_distribution
    )
    
    if len(train_dataset) == 0:
        print(f"Warning: Client {client_id} has no training data")
        return None, None, None, None
    
    actual_batch_size = min(config.batch_size, len(train_dataset))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=actual_batch_size,
        shuffle=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    print(f"Client {client_id}: {len(train_dataset)} train, {len(test_dataset)} test samples")
    
    return train_loader, test_loader, train_dataset.get_input_dim(), train_dataset.get_num_classes()