"""Download original MovieLens or public ratings mirrors; verify official split bytes."""
from pathlib import Path
import argparse, concurrent.futures, hashlib, json, sys, urllib.request, zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dpt_sr.data import ROOT, prepare, raw_frame, leave_one_out, processed_paths, verify_processed, sha256

def fetch(url, path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): return path
    part=path.with_suffix(path.suffix+'.part')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url,timeout=45) as src, part.open('wb') as out:
                while block:=src.read(1<<20):out.write(block)
            part.replace(path);return path
        except OSError:
            if attempt==2:raise

def run(dataset,data_dir,source,raw_dir):
    if all(p.exists() for p in processed_paths(dataset,data_dir)):
        return verify_processed(dataset,data_dir)
    if source=='public':return prepare(dataset,data_dir,raw_dir)
    raw_dir=Path(raw_dir);raw_dir.mkdir(parents=True,exist_ok=True)
    if dataset=='ml1m':
        urls=['https://raw.githubusercontent.com/vandit15/Movielens-Data/master/ml-1m/ratings.dat']
        files=[fetch(urls[0],raw_dir/'ml1m_ratings.dat')]
        if sha256(files[0])!='506d64ca44484487c11dc2d9a28de5c54948213e6b96285e298afe28d6ea4e0f':
            raise ValueError('MovieLens-1M mirror changed')
        archive=raw_dir/'ml1m_mirror.zip'
        with zipfile.ZipFile(archive,'w') as z:z.write(files[0],'ml-1m/ratings.dat')
    else:
        urls=[f'https://raw.githubusercontent.com/ChicagoBoothML/DATA___MovieLens___20M/master/ratings{i:02}.csv' for i in range(1,11)]
        files=[raw_dir/'ml20m_parts'/f'ratings{i:02}.csv' for i in range(1,11)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for path in pool.map(lambda x:fetch(*x),zip(urls,files)):print('Downloaded',path.name,flush=True)
        archive=raw_dir/'ml20m_mirror.zip'
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as z:
            with z.open('ml-20m/ratings.csv','w',force_zip64=True) as out:
                for i,path in enumerate(files):
                    with path.open('rb') as f:
                        if i:f.readline()
                        while block:=f.read(1<<20):out.write(block)
    print('Preprocessing',dataset,flush=True)
    frames=leave_one_out(raw_frame(dataset,archive))
    for frame,path in zip(frames,processed_paths(dataset,data_dir)):
        path.parent.mkdir(parents=True,exist_ok=True)
        frame.to_csv(path,index=False,lineterminator='\n')
    verified=verify_processed(dataset,data_dir)
    record={'dataset':dataset,'source':'public ratings mirror, not the official raw ZIP',
        'raw_files':[{'url':u,'bytes':p.stat().st_size,'sha256':sha256(p)} for u,p in zip(urls,files)],
        'processed':verified,'rows':[len(f) for f in frames]}
    dest=ROOT/'evidence/external_data'/f'{dataset}.json';dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
    return verified

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=['ml1m','ml20m'],required=True)
    p.add_argument('--source',choices=['public','mirror'],default='public')
    p.add_argument('--data-dir',type=Path,default=ROOT/'data');p.add_argument('--raw-dir',type=Path,default=ROOT/'review_sources')
    a=p.parse_args();run(a.dataset,a.data_dir,a.source,a.raw_dir)
