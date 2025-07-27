import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MultivariateDatasetBenchmark(Dataset):
    def __init__(
        self,
        seq_len,
        input_token_len,
        output_token_len,
        pred_len,
        data_path,
        flag,
        scale=True,
    ):
        self.seq_len = seq_len
        self.input_token_len = input_token_len
        self.output_token_len = output_token_len
        self.pred_len = pred_len
        self.token_num = self.seq_len // self.input_token_len
        self.flag = flag
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]
        self.dataset_file_path = data_path
        self.data_type = os.path.basename(self.dataset_file_path).lower()
        self.scale = scale
        self.mean = None
        self.std = None
        self.__read_data__()

    def __read_data__(self):
        if self.dataset_file_path.endswith(".csv"):
            df_raw = pd.read_csv(self.dataset_file_path)
        elif self.dataset_file_path.endswith(".txt"):
            df_raw = []
            with open(self.dataset_file_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip("\n").split(",")
                    data_line = np.stack([float(i) for i in line])
                    df_raw.append(data_line)
            df_raw = np.stack(df_raw, 0)
            df_raw = pd.DataFrame(df_raw)
        elif self.dataset_file_path.endswith(".npz"):
            data = np.load(self.dataset_file_path, allow_pickle=True)
            data = data["data"][:, :, 0]
            df_raw = pd.DataFrame(data)
        elif self.dataset_file_path.endswith(".npy"):
            data = np.load(self.dataset_file_path)
            df_raw = pd.DataFrame(data)
        elif self.dataset_file_path.endswith(".h5"):
            import h5py

            f = h5py.File(self.dataset_file_path, "r")
            try:
                df_raw = pd.DataFrame(f["df"]["block0_values"])
            except:
                df_raw = pd.DataFrame(f["speed"]["block0_values"])
        else:
            raise ValueError("Unknown data format: {}".format(self.dataset_file_path))

        if "etth" in self.data_type:
            border1s = [
                0,
                12 * 30 * 24 - self.seq_len,
                12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
            ]
            border2s = [
                12 * 30 * 24,
                12 * 30 * 24 + 4 * 30 * 24,
                12 * 30 * 24 + 8 * 30 * 24,
            ]
        elif "ettm" in self.data_type:
            border1s = [
                0,
                12 * 30 * 24 * 4 - self.seq_len,
                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len,
            ]
            border2s = [
                12 * 30 * 24 * 4,
                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
                12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
            ]
        else:
            data_len = len(df_raw)
            num_train = int(data_len * 0.7)
            num_test = int(data_len * 0.2)
            num_vali = data_len - num_train - num_test
            border1s = [0, num_train - self.seq_len, data_len - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, data_len]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if isinstance(df_raw[df_raw.columns[0]][2], str):
            data = df_raw[df_raw.columns[1:]].values
        else:
            data = df_raw.values

        if self.scale:
            train_data = data[border1s[0] : border2s[0]]
            self.mean = np.mean(train_data, axis=0)
            self.std = np.std(train_data, axis=0) + 1e-5
            data = (data - self.mean) / self.std

            self.mean = np.mean(train_data)
            self.std = np.std(train_data) + 1e-5

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        num_channel = self.data_x.shape[1]
        s_begin = index
        s_end = s_begin + self.seq_len
        if self.flag == "train":
            r_begin = s_begin + self.input_token_len
            r_end = s_end + self.output_token_len
        else:
            r_begin = s_end
            r_end = r_begin + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        loss_mask = np.ones((self.token_num, num_channel))
        y_mask = np.zeros((self.output_token_len, num_channel))
        y_mask[: self.pred_len, :] = 1.0
        return seq_x, seq_y, loss_mask, y_mask

    def __len__(self):
        if self.flag == "train":
            return len(self.data_x) - self.seq_len - self.output_token_len + 1
        else:
            return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        if isinstance(self.mean, np.ndarray) and isinstance(self.std, np.ndarray):
            self.mean = torch.from_numpy(self.mean).float().to(data.device)
            self.std = torch.from_numpy(self.std).float().to(data.device)
        return data * self.std + self.mean


class FinetuneDatasetBenchmark(Dataset):
    def __init__(
        self,
        seq_len,
        input_token_len,
        output_token_len,
        pred_len,
        data_path,
        flag,
        scale=True,
    ):
        self.seq_len = seq_len
        self.input_token_len = input_token_len
        self.output_token_len = output_token_len
        self.pred_len = pred_len
        self.token_num = self.seq_len // self.input_token_len
        self.flag = flag
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]
        self.dataset_file_path = data_path
        self.data_type = os.path.basename(self.dataset_file_path).lower()
        self.scale = scale
        self.mean = None
        self.std = None
        self.__read_data__()

    def __read_data__(self):
        if self.dataset_file_path.endswith(".csv"):
            df_raw = pd.read_csv(self.dataset_file_path)
        elif self.dataset_file_path.endswith(".txt"):
            df_raw = []
            with open(self.dataset_file_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip("\n").split(",")
                    data_line = np.stack([float(i) for i in line])
                    df_raw.append(data_line)
            df_raw = np.stack(df_raw, 0)
            df_raw = pd.DataFrame(df_raw)
        elif self.dataset_file_path.endswith(".npz"):
            data = np.load(self.dataset_file_path, allow_pickle=True)
            data = data["data"][:, :, 0]
            df_raw = pd.DataFrame(data)
        elif self.dataset_file_path.endswith(".npy"):
            data = np.load(self.dataset_file_path)
            df_raw = pd.DataFrame(data)
        elif self.dataset_file_path.endswith(".h5"):
            import h5py

            f = h5py.File(self.dataset_file_path, "r")
            try:
                df_raw = pd.DataFrame(f["df"]["block0_values"])
            except:
                df_raw = pd.DataFrame(f["speed"]["block0_values"])
        else:
            raise ValueError("Unknown data format: {}".format(self.dataset_file_path))

        if "etth" in self.data_type:
            border1s = [
                0,
                12 * 30 * 24 - self.seq_len,
                12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
            ]
            border2s = [
                12 * 30 * 24,
                12 * 30 * 24 + 4 * 30 * 24,
                12 * 30 * 24 + 8 * 30 * 24,
            ]
        elif "ettm" in self.data_type:
            border1s = [
                0,
                12 * 30 * 24 * 4 - self.seq_len,
                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len,
            ]
            border2s = [
                12 * 30 * 24 * 4,
                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
                12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
            ]
        else:
            data_len = len(df_raw)
            num_train = int(data_len * 0.7)
            num_test = int(data_len * 0.2)
            num_vali = data_len - num_train - num_test
            border1s = [0, num_train - self.seq_len, data_len - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, data_len]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if isinstance(df_raw[df_raw.columns[0]][2], str):
            data = df_raw[df_raw.columns[1:]].values
        else:
            data = df_raw.values
        if self.scale:
            train_data = data[border1s[0] : border2s[0]]

            self.mean = np.mean(train_data, axis=0)
            self.std = np.std(train_data, axis=0) + 1e-5
            data = (data - self.mean) / self.std

            self.mean = np.mean(train_data)
            self.std = np.std(train_data) + 1e-5

        self.data_x = data[border1:border2].astype(float)
        self.data_y = data[border1:border2].astype(float)

        self.n_var = self.data_x.shape[-1]
        if self.flag == "train":
            self.n_timepoint = (
                len(self.data_x) - self.seq_len - self.output_token_len + 1
            )
        else:
            self.n_timepoint = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        feat_id = index // self.n_timepoint
        s_begin = index % self.n_timepoint
        s_end = s_begin + self.seq_len

        if self.flag == "train":
            r_begin = s_begin + self.input_token_len
            r_end = s_end + self.output_token_len
        else:
            r_begin = s_end
            r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end, feat_id]
        seq_y = self.data_y[r_begin:r_end, feat_id]
        loss_mask = np.ones(self.token_num, dtype=np.int32)
        y_mask = np.zeros((self.output_token_len))
        y_mask[: self.pred_len] = 1.0
        return seq_x, seq_y, loss_mask, y_mask

    def __len__(self):
        return int(self.n_var * self.n_timepoint)

    def inverse_transform(self, data):
        if isinstance(self.mean, np.ndarray) and isinstance(self.std, np.ndarray):
            self.mean = torch.from_numpy(self.mean).float().to(data.device)
            self.std = torch.from_numpy(self.std).float().to(data.device)
        return data * self.std + self.mean
