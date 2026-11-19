"""Pair-support penalty and mass-tempered history weighting."""
from dataclasses import dataclass,asdict,replace
from pathlib import Path
import json,time
import numpy as np
from scipy import sparse
from scipy.special import expit
from capacity_probes.counts import OffsetCache
from capacity_probes.models.pctm import count_weights
from .model import array_bytes
from .serialization import save,load


@dataclass(frozen=True)
class CalibrationConfig:
    pair_penalty: float = 0.5
    exposure_power: float = 0.25

    def __post_init__(self):
        if not np.isfinite(self.pair_penalty) or self.pair_penalty<0:
            raise ValueError('pair_penalty must be finite and nonnegative')
        if not np.isfinite(self.exposure_power) or not 0<=self.exposure_power<=1:
            raise ValueError('exposure_power must be in [0,1]')


@dataclass
class CalibratedModel:
    base: object
    calibration_config: CalibrationConfig
    history_scale: np.ndarray

    def __getattr__(self,name):
        return getattr(self.base,name)


def calibrate_history(history,scale):
    weighted=(history@sparse.diags(scale)).tocsr()
    mass=np.asarray(weighted.sum(axis=1)).ravel()
    if np.any(mass<=0):raise ValueError('History must have positive total weight')
    return (sparse.diags(1/mass)@weighted).tocsr()


def array_bytes_calibrated(model):
    return array_bytes(model.base)+model.history_scale.nbytes


def save_calibrated(model,path):
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    save(model.base,path/'base')
    (path/'calibration.json').write_text(json.dumps({'format':'DPT-SR-calibration-1','calibration_config':asdict(model.calibration_config)},indent=2),encoding='utf-8')
    np.savez_compressed(path/'history_scale.npz',history_scale=model.history_scale)


def load_calibrated(path):
    path=Path(path);meta=json.loads((path/'calibration.json').read_text(encoding='utf-8'))
    if meta['format']!='DPT-SR-calibration-1':raise ValueError('Unsupported calibration format')
    base=load(path/'base')
    with np.load(path/'history_scale.npz',allow_pickle=False) as a:scale=a['history_scale']
    if scale.shape!=(len(base.catalog),) or not np.all(np.isfinite(scale)) or np.any(scale<=0):
        raise ValueError('Invalid history-scale array')
    return CalibratedModel(base,CalibrationConfig(**meta['calibration_config']),scale)
