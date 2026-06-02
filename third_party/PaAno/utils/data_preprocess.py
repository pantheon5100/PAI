import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.cluster import KMeans, MiniBatchKMeans
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Data Load
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            print(f"Empty file: {file_path}")
            return [], []
        values = df.to_numpy(copy=False)
        data = values[:, :-1]
        if data.ndim == 2 and data.shape[1] == 1:
            data = data[:, 0]
        labels = values[:, -1]
        return data, labels
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return [], []


def _extract_train_end(file_path):
    file_name_parts = os.path.basename(file_path).split('_')
    for idx, part in enumerate(file_name_parts[:-1]):
        if part == 'tr':
            train_end = os.path.splitext(file_name_parts[idx + 1])[0]
            if train_end.isdigit():
                return int(train_end)
            break
    raise ValueError(f"Invalid file format or missing 'tr_' in: {file_path}")


def load_and_split_data(file_path):
    try:
        data, labels = load_data(file_path)
        if len(data) == 0 or len(labels) == 0:
            print(f"Empty data or labels for file: {file_path}")
            return [], [], [], []
        train_end = _extract_train_end(file_path)
        train_data = data[:train_end]
        train_labels = labels[:train_end]
        test_data = data[train_end:]
        test_labels = labels[train_end:]

        return train_data, train_labels, test_data, test_labels
    except IsADirectoryError:
        print(f"Skipped directory: {file_path}")
        return None, None, None, None
    except ValueError as e:
        print(f"Error processing file {file_path}: {e}")
        return None, None, None, None


def _patch_start_indices(total_length, patch_size, stride):
    if total_length < patch_size:
        raise ValueError(f"Data length {total_length} is less than patch size {patch_size}.")
    return torch.arange(0, total_length - patch_size + 1, stride, dtype=torch.long)


def patch_window_count(total_length, patch_size, stride):
    if total_length < patch_size:
        raise ValueError(f"Data length {total_length} is less than patch size {patch_size}.")
    return ((total_length - patch_size) // stride) + 1


def create_patch_tensor_and_indices(data, patch_size, stride):
    array = np.asarray(data)
    indices = _patch_start_indices(len(array), patch_size, stride)

    if array.ndim == 1:
        windows = sliding_window_view(np.ascontiguousarray(array), patch_size)[::stride]
        tensor = torch.from_numpy(np.ascontiguousarray(windows)).float().unsqueeze(1).contiguous()
        return tensor, indices

    if array.ndim == 2:
        base = torch.as_tensor(np.ascontiguousarray(array).T, dtype=torch.float32)
        tensor = base.unfold(1, patch_size, stride).permute(1, 0, 2).contiguous()
        return tensor, indices

    raise ValueError(f"Expected 1D or 2D time series, got shape {tuple(array.shape)}")


def preprocess_to_patches(data, patch_size, stride):
    tensor, _ = create_patch_tensor_and_indices(data, patch_size, stride)
    return tensor


class _tsdataset(Dataset):
    def __init__(self, data, indices=None):
        # indice means relative order among patches
        if torch.is_tensor(data):
            self.data = data.float().contiguous()
        else:
            self.data = torch.from_numpy(np.asarray(data)).float().contiguous()
        if indices is not None:
            if torch.is_tensor(indices):
                self.indices = indices.long().view(-1, 1).contiguous()
            else:
                self.indices = torch.from_numpy(np.asarray(indices)).long().view(-1, 1).contiguous()
        else:
            self.indices = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
            # x: (L,) or (L, C) or (C, L)
        if x.ndim == 1:          # (L,) -> (1, L)
            x = x.unsqueeze(0).contiguous()
       
        if self.indices is not None:
            return x, self.indices[idx]
        return x, torch.tensor([idx])

class PatchCreator:
    def __init__(self, L, s, random_seed=None):
        self.L = L
        self.s = s
        if random_seed is not None:
            torch.manual_seed(random_seed)

    def create_patches(self, data):
        if torch.is_tensor(data):
            data = data.detach().cpu().contiguous().numpy()
        elif not isinstance(data, (list, np.ndarray)):
            raise ValueError("Data must be a list or numpy array.")
        patch_tensor, indices = create_patch_tensor_and_indices(np.asarray(data), self.L, self.s)
        return patch_tensor, indices

    def create_dataloaders_from_patches(self, train_patches, train_indices, test_patches, test_indices, batch_size=512):
        pin_memory = torch.cuda.is_available()
        train_loader = DataLoader(
            _tsdataset(train_patches, indices=train_indices),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            _tsdataset(test_patches, indices=test_indices),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
        )
        return train_loader, test_loader

    def create_dataloaders(self, train_data, full_data, test_labels, batch_size=512):
        full_patches, full_indices = self.create_patches(full_data)
        train_patch_count = patch_window_count(len(train_data), self.L, self.s)
        train_patches = full_patches[:train_patch_count]
        train_indices = full_indices[:train_patch_count]
        train_loader, test_loader = self.create_dataloaders_from_patches(
            train_patches,
            train_indices,
            full_patches,
            full_indices,
            batch_size=batch_size,
        )

        true_test_labels = test_labels

        return train_loader, test_loader, true_test_labels



# cited from https://github.com/TheDatumOrg/TSB-AD/blob/main/TSB_AD/utils/slidingWindows.py
from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf

# determine sliding window (period) based on ACF
def find_length_rank(data, rank=1):
    data = data.squeeze()
    if len(data.shape)>1: return 100 #0->100
    if rank==0: return 1
    data = data[:min(20000, len(data))]
    
    base = 3
    auto_corr = acf(data, nlags=400, fft=True)[base:]
    
    # plot_acf(data, lags=400, fft=True)
    # plt.xlabel('Lags')
    # plt.ylabel('Autocorrelation')
    # plt.title('Autocorrelation Function (ACF)')
    # plt.savefig('/data/liuqinghua/code/ts/TSAD-AutoML/AutoAD_Solution/candidate_pool/cd_diagram/ts_acf.png')

    local_max = argrelextrema(auto_corr, np.greater)[0]

    # print('auto_corr: ', auto_corr)
    # print('local_max: ', local_max)

    try:
        # max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        sorted_local_max = np.argsort([auto_corr[lcm] for lcm in local_max])[::-1]    # Ascending order
        max_local_max = sorted_local_max[0]     # Default
        if rank == 1: max_local_max = sorted_local_max[0]
        if rank == 2: 
            for i in sorted_local_max[1:]: 
                if i > sorted_local_max[0]: 
                    max_local_max = i 
                    break
        if rank == 3:
            for i in sorted_local_max[1:]: 
                if i > sorted_local_max[0]: 
                    id_tmp = i
                    break
            for i in sorted_local_max[id_tmp:]:
                if i > sorted_local_max[id_tmp]: 
                    max_local_max = i           
                    break
        # print('sorted_local_max: ', sorted_local_max)
        # print('max_local_max: ', max_local_max)
        if local_max[max_local_max]<3 or local_max[max_local_max]>300:
            return 125
        return local_max[max_local_max]+base
    except:
        return 125
    

# determine sliding window (period) based on ACF, Original version
def find_length(data):
    if len(data.shape)>1:
        return 0
    data = data[:min(20000, len(data))]
    
    base = 3
    auto_corr = acf(data, nlags=400, fft=True)[base:]
    
    
    local_max = argrelextrema(auto_corr, np.greater)[0]
    try:
        max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
        if local_max[max_local_max]<3 or local_max[max_local_max]>300:
            return 125
        return local_max[max_local_max]+base
    except:
        return 125
