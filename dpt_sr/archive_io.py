"""Atomic, CRC-checked writing of portable NumPy result archives."""
from pathlib import Path
import io,zipfile
import numpy as np

def save_npz_atomic(path,**arrays):
    path=Path(path);buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);raw=buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        if z.testzip() is not None:raise IOError('Incomplete result archive')
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_bytes(raw);temporary.replace(path)
