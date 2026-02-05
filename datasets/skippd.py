from ast import Yield
from datetime import datetime, timezone
import pvlib
import time
from datasets import load_dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from pathlib import Path
from torch.utils.data import Dataset
from typing import Literal
from datetime import datetime, timedelta
import pandas as pd
import random
import torch

def clearsky_irradiance(timestamps, lat, lon):
    times = pd.DatetimeIndex(timestamps)
    location = pvlib.location.Location(lat, lon)
    cs = location.get_clearsky(times, model='ineichen')
    cs_irradiance = {}
    cs_irradiance['dni'] = cs['dni'].to_numpy()
    cs_irradiance['ghi'] = cs['ghi'].to_numpy()
    cs_irradiance['dhi'] = cs['dhi'].to_numpy()
    return cs_irradiance

def estimate_cs_pv(P, cs, cos_zenith, solpos, tilt, azimuth):
    """
    用高分位晴天样本拟合比例系数 alpha
    P ≈ alpha * (dni * cos_zenith + dhi)
    """
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        dni=cs["dni"],
        ghi=cs["ghi"],
        dhi=cs["dhi"],
        solar_zenith=solpos["zenith"],
        solar_azimuth=solpos["azimuth"],
        albedo=0.25,
        model="isotropic"
    )
    poa_cs = poa["poa_global"].to_numpy()  # W/m^2

    mask = (cs['dni'] > 200) & (cos_zenith > 0.2)
    y = P[mask]

    # 用 95% 分位拟合上包络
    alpha = np.percentile(y / (poa_cs[mask] + 1e-6), 75)
    print('----------- ', alpha)
    
    P_cs = alpha * poa_cs
    return P_cs

def estimate_cs_pv_walpha(alpha, cs, cos_zenith, solpos, tilt, azimuth):
    """
    用高分位晴天样本拟合比例系数 alpha
    P ≈ alpha * (dni * cos_zenith + dhi)
    """
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        dni=cs["dni"],
        ghi=cs["ghi"],
        dhi=cs["dhi"],
        solar_zenith=solpos["zenith"],
        solar_azimuth=solpos["azimuth"],
        albedo=0.25,
        model="isotropic"
    )
    poa_cs = poa["poa_global"].to_numpy()  # W/m^2
    P_cs = alpha * poa_cs
    return P_cs

def zenith_azimuth_from_datetime(
    timestamps,
    latitude: float,
    longitude: float,
):
    """
    timestamps: pandas DatetimeIndex 或 list of datetime
                ⚠️ 必须是 tz-aware（带 timezone 信息）
    latitude, longitude: 站点经纬度（度）

    return dict:
        cos_zenith: [N]   ⭐ 最重要
        sin_azimuth: [N] （可选）
        cos_azimuth: [N] （可选）
        zenith_deg, azimuth_deg
    """

    # 确保是 DatetimeIndex
    times = pd.DatetimeIndex(timestamps)

    # 强制检查是否带 timezone
    if times.tz is None:
        raise ValueError("timestamps must be timezone-aware (tz info required)")

    # 计算太阳位置（pvlib 自动处理时区 / UTC / equation of time 等）
    solpos = pvlib.solarposition.get_solarposition(
        time=times,
        latitude=latitude,
        longitude=longitude
    )

    zenith = solpos["zenith"].to_numpy()      # 天顶角（度）
    azimuth = solpos["azimuth"].to_numpy()    # 方位角（度）

    # 核心特征
    cos_zenith = np.clip(np.cos(np.deg2rad(zenith)), 0.0, 1.0)

    # 可选增强特征
    sin_azimuth = np.sin(np.deg2rad(azimuth))
    cos_azimuth = np.cos(np.deg2rad(azimuth))

    return {
        "cos_zenith": cos_zenith,      # ⭐⭐⭐⭐⭐ 最重要特征
        "sin_azimuth": sin_azimuth,    # 可选
        "cos_azimuth": cos_azimuth,    # 可选
        "zenith_deg": zenith,
        "azimuth_deg": azimuth,
    }

class SkippdDataset(Dataset):
    def __init__(
        self,
        lon = -122.174, # longitude in degrees
        lat = 34.427, # latitude in degrees
        split: Literal["train", "test"] = "train",
        sample_num: int = 100000,
        alpha: float = None,
    ):
        dataset = load_dataset("solarbench/SKIPPD", split=split, download_mode="reuse_dataset_if_exists")
        self.sample_num = sample_num

        pv = np.array(dataset['pv'])

        JDs = []
        for dt in dataset['time']:
            # Convert timezone-aware datetime to UTC
            if dt.tzinfo is not None:
                dt_utc = dt.astimezone(timezone.utc)
            else:
                # If naive datetime, assume it's already in UTC or convert as needed
                dt_utc = dt.replace(tzinfo=timezone.utc)
            unix_timestamp = int(dt_utc.timestamp())
            julian_date = pvlib.spa.julian_day(unix_timestamp)
            JDs.append(julian_date)

        JDs = np.asarray(JDs)
        zenith_azimuth = zenith_azimuth_from_datetime( dataset['time'], lat, lon)
        cs = clearsky_irradiance(dataset['time'], lat, lon)
        solpos = pvlib.solarposition.get_solarposition(dataset['time'], lat, lon)
        
        if split == "train":
            self.pv_cs = estimate_cs_pv(np.array(dataset['pv']), cs, zenith_azimuth['cos_zenith'], solpos, tilt=22.5, azimuth=195)
        else:
            self.pv_cs = estimate_cs_pv_walpha(alpha, cs, zenith_azimuth['cos_zenith'], solpos, tilt=22.5, azimuth=195)
        
        self._dataset = dataset
        self.pv = pv
        self.JDs = JDs
        self.zenith_azimuth = zenith_azimuth
        self.timestamps = dataset['time']
        self.residual = (self.pv - self.pv_cs + 2.8) / 5.6  # z-score normalization

        if split == "train":
            self.samples = []
            for _ in range(sample_num):
                idx = random.randint(0, len(self.pv) - 1)
                t_current = JDs[idx]
                t_prev = t_current - 15 / 24 / 60
                t_pred = t_current + 15 / 24 / 60
                indices_window = np.where((self.JDs >= t_prev) & (self.JDs <= t_current))[0][-15:]
                index_pred = np.where(np.abs(self.JDs - t_pred) < 10 / 24 / 3600)[0]
                if index_pred.shape[0] == 1:
                    self.samples.append((indices_window.copy(), index_pred.copy()))
        else:  # split == "test": use all valid samples, pad in __getitem__ if < 15
            self.samples = []
            for idx in range(len(self.pv)):
                t_current = JDs[idx]
                t_prev = t_current - 15 / 24 / 60
                t_pred = t_current + 15 / 24 / 60
                indices_window = np.where((self.JDs >= t_prev) & (self.JDs <= t_current))[0][-15:]
                index_pred = np.where(np.abs(self.JDs - t_pred) < 10 / 24 / 3600)[0]
                if index_pred.shape[0] == 1 and indices_window.shape[0] == 15:
                    self.samples.append((indices_window.copy(), index_pred.copy()))    

        
        de = 0
        # plt.plot(pv[de*650:(de+10)*650],'-')
        plt.plot(self.pv_cs[de*650:(de+10)*650]-pv[de*650:(de+10)*650],'-r')
        plt.savefig('skippd.png')
        plt.close()
        

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        indices_window, index_pred = self.samples[idx]
        T = len(indices_window)

        # Build tensors from valid indices
        residual = torch.tensor(self.residual[indices_window], dtype=torch.float32)
        cos_zenith = torch.tensor(self.zenith_azimuth['cos_zenith'][indices_window], dtype=torch.float32)
        cos_azimuth = torch.tensor(self.zenith_azimuth['cos_azimuth'][indices_window], dtype=torch.float32)
        sin_azimuth = torch.tensor(self.zenith_azimuth['sin_azimuth'][indices_window], dtype=torch.float32)
        imgs = np.stack([np.array(self._dataset['image'][i]) for i in indices_window])
        image = torch.tensor(imgs, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0

        # Pad with zeros on the left if T < 15
        if T < 15:
            pad = 15 - T
            residual = torch.cat([torch.zeros(pad, dtype=torch.float32), residual], dim=0)
            cos_zenith = torch.cat([torch.zeros(pad, dtype=torch.float32), cos_zenith], dim=0)
            cos_azimuth = torch.cat([torch.zeros(pad, dtype=torch.float32), cos_azimuth], dim=0)
            sin_azimuth = torch.cat([torch.zeros(pad, dtype=torch.float32), sin_azimuth], dim=0)
            image = torch.cat([torch.zeros(pad, *image.shape[1:], dtype=image.dtype), image], dim=0)

        if index_pred.shape[0]==0:
            target = torch.tensor([0.0], dtype=torch.float32)
            pv_target = torch.tensor([0.0], dtype=torch.float32)
            pv_cs_target = torch.tensor([0.0], dtype=torch.float32)
        else:
            target = torch.tensor(self.residual[index_pred], dtype=torch.float32)
            pv_target = torch.tensor(self.pv[index_pred], dtype=torch.float32)
            pv_cs_target = torch.tensor(self.pv_cs[index_pred], dtype=torch.float32)

        return {
            'residual': residual,
            'cos_zenith': cos_zenith,
            'cos_azimuth': cos_azimuth,
            'sin_azimuth': sin_azimuth,
            'image': image,
            'pv_target': pv_target,
            'pv_cs_target': pv_cs_target,
            'target': target,
        }


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    print("Testing SkippdDataset...")
    
    # Test test dataset
    print("\n=== Creating test dataset ===")
    test_dataset = SkippdDataset(
        lon=-122.174,
        lat=34.427,
        split="train",
        alpha=0.02717255163581315,
    )
    print(f"Test dataset length: {len(test_dataset)}")

    # Test DataLoader
    print("\n=== Testing DataLoader ===")
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )
    print(f"Test batches: {len(test_loader)}")

    # Load data from test_loader
    print("\n=== Loading batches from test_loader ===")
    for batch_idx, batch in enumerate(test_loader):
        print(f"Batch {batch_idx}: keys={list(batch.keys())}")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"  {k}: {type(v).__name__}")
    print("\n=== test_loader check done ===")
