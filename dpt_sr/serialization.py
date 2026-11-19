import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from scipy import sparse
from capacity_probes.models.pctm import PCTMConfig
from .model import DPTConfig,DPTModel


def save(model,path):
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    (path/'metadata.json').write_text(json.dumps({'format':1,'config':asdict(model.config),
        'pctm_config':asdict(model.pctm_config),'max_history':model.max_history,'diagnostics':model.diagnostics},indent=2),encoding='utf-8')
    np.savez_compressed(path/'arrays.npz',catalog=model.catalog,theta=model.theta,
                       row_background=model.row_background,log_popularity=model.log_popularity)
    sparse.save_npz(path/'evidence.npz',model.evidence)


def load(path):
    path=Path(path);meta=json.loads((path/'metadata.json').read_text(encoding='utf-8'))
    if meta['format']!=1:raise ValueError('Unsupported model format')
    with np.load(path/'arrays.npz',allow_pickle=False) as a:
        return DPTModel(DPTConfig(**meta['config']),PCTMConfig(**meta['pctm_config']),a['catalog'],
            sparse.load_npz(path/'evidence.npz'),a['row_background'],a['log_popularity'],a['theta'],
            meta['max_history'],meta['diagnostics'])
