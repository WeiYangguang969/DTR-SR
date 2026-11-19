"""One independent process, one verified split, exact full-catalog top-10."""
from pathlib import Path
import argparse, copy, hashlib, json, sys, time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataclasses import asdict
import numpy as np
from threadpoolctl import threadpool_limits
from dpt_sr.data import load_split, leave_one_out
from dpt_sr.model import DPTConfig, cache_for
from dpt_sr.calibration import CalibrationConfig, calibrate_history
from dpt_sr.full_model import (CompletionConfig, fit_completion, recommend_completion,
    array_bytes_completion, save_completion, load_completion, score_completion)
from dpt_sr.runtime import environment, peak_rss_mib, directory_bytes, write_json
from capacity_probes.sequences import SequenceStore, EvaluationSet
from capacity_probes.models.pctm import build_pctm, recommend_pctm, kernel_weights
from capacity_probes.models.seqrules import build_seqrules, recommend_seqrules
from capacity_probes.metrics import official_metrics

ROOT = Path(__file__).resolve().parents[1]

def specification(dataset, variant, strength=None, fanout=None):
    spec = copy.deepcopy(json.loads((ROOT/'configs/final_model.json').read_text())[dataset])
    if variant == 'no_direction': spec['dpt']['kappa'] = 0.
    elif variant == 'no_history': spec['calibration']['exposure_power'] = 0.
    elif variant == 'no_path': spec['completion']['strength'] = 0.
    elif variant == 'neutral_prior': spec['dpt']['prior'] = 'neutral'
    elif variant == 'no_gate': spec['calibration']['pair_penalty'] = 0.
    elif variant == 'renormalized': spec['completion']['renormalize_paths'] = True
    elif variant == 'opposite_mask': spec['completion']['exclude_observed'] = not spec['completion']['exclude_observed']
    elif variant not in ('full', 'pctm', 'seqrules'): raise ValueError(variant)
    if strength is not None: spec['completion']['strength'] = strength
    if fanout is not None: spec['completion']['fanout'] = fanout
    return spec

def run(args):
    import rectools
    if rectools.__version__ != '0.13.0': raise RuntimeError('RecTools 0.13.0 is required')
    out = args.output
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    spec = specification(args.dataset, args.variant, args.strength, args.fanout)
    train, held, verified = load_split(args.dataset, args.data_dir, args.split)
    if args.split == 'valid':
        for _ in range(args.depth-1): train, held = leave_one_out(train)
    elif args.depth != 1: raise ValueError('Test depth must equal one')
    store = SequenceStore.from_frame(train)
    ev = EvaluationSet.from_holdout(store, held)
    cache = cache_for(store, args.cache_dir)
    if cache.exists(): raise RuntimeError('Cold-cache run requires a new cache directory')
    with threadpool_limits(limits=1):
        tick = time.perf_counter()
        if args.variant == 'pctm':
            model = build_pctm(store, cache, spec['pctm'])
            e = model.evidence
            array_mib = (e.data.nbytes+e.indices.nbytes+e.indptr.nbytes+model.log_popularity.nbytes)/2**20
        elif args.variant == 'seqrules':
            model = build_seqrules(store, cache, spec['seqrules'])
            e = model.rules
            array_mib = (e.data.nbytes+e.indices.nbytes+e.indptr.nbytes)/2**20
        else:
            model = fit_completion(store, spec['pctm'], DPTConfig(**spec['dpt']),
                CalibrationConfig(**spec['calibration']), CompletionConfig(**spec['completion']),
                spec['max_history'], args.cache_dir)
            array_mib = array_bytes_completion(model)/2**20
        fit_seconds = time.perf_counter()-tick
        tick = time.perf_counter()
        if args.variant == 'pctm': local, reco = recommend_pctm(model, ev, spec['max_history'])
        elif args.variant == 'seqrules': local, reco = recommend_seqrules(model, ev, spec['max_history'])
        else: local, reco = recommend_completion(model, ev)
        rank_seconds = time.perf_counter()-tick
        tick = time.perf_counter()
        official = official_metrics(reco, train, held)
        metric_seconds = time.perf_counter()-tick
        np.testing.assert_allclose([local.ndcg10, local.recall10],
            [official['ndcg@10'], official['recall@10']], atol=1e-12, rtol=0)
        hits = reco.merge(held[['user_id','item_id']], on=['user_id','item_id'], how='inner')
        ranks = np.zeros(len(ev.users), np.int16)
        ranks[np.searchsorted(ev.users, hits.user_id.to_numpy())] = hits['rank'].to_numpy()
        keep = ranks > 0
        ndcg = np.zeros(len(ev.users)); ndcg[keep] = 1/np.log2(ranks[keep].astype(float)+1)
        np.testing.assert_allclose(ndcg.mean(), official['ndcg@10'], atol=1e-12, rtol=0)
        targets = np.asarray([t[0] for t in ev.targets])
        per = dict(user_id=ev.users, ndcg10=ndcg, recall10=keep.astype(float), rank_in_top10=ranks,
            history_length=store.ends[ev.groups]-store.starts[ev.groups],
            target_frequency=store.item_frequency()[targets], target_item=store.catalog[targets])
        serialization = False
        if args.variant not in ('pctm','seqrules'):
            h = ev.history_matrix(spec['max_history'], lambda n: kernel_weights(spec['pctm']['kernel'], n))
            if spec['calibration']['exposure_power'] > 0: h = calibrate_history(h, model.history_scale)
            rows = np.repeat(np.arange(len(ev.users)), np.diff(h.indptr))
            def reachable(evidence):
                values = np.asarray(evidence[h.indices, targets[rows]]).ravel()*h.data
                return np.bincount(rows, weights=values, minlength=len(ev.users)) > 0
            direct, added = reachable(model.evidence), reachable(model.completion_evidence)
            per.update(direct_reachable=direct, completion_reachable=added, reachable=direct|added)
            if args.save_model:
                save_completion(model, out/'model')
                loaded = load_completion(out/'model')
                np.testing.assert_array_equal(score_completion(model,h[:256]).toarray(),
                    score_completion(loaded,h[:256]).toarray())
                serialization = True
        np.savez_compressed(out/'per_user.npz', **per)
        if args.save_model: reco.to_csv(out/'recommendations.csv', index=False, lineterminator='\n')
    result = dict(dataset=args.dataset, variant=args.variant, split=args.split, depth=args.depth,
        spec=spec, official_metrics=official, metrics=asdict(local), data_sha256=verified,
        fit_rows=len(train), train_users=int(train.user_id.nunique()), holdout_rows=len(held),
        catalog_items=store.num_items, diagnostics=getattr(model,'diagnostics',{}),
        resources=dict(fit_seconds=fit_seconds, ranking_seconds=rank_seconds,
            official_metric_seconds=metric_seconds, elapsed_seconds=time.perf_counter()-start,
            peak_process_rss_mib=peak_rss_mib(), model_array_mib=array_mib,
            cache_mib=directory_bytes(cache)/2**20, cache_existed_at_start=False,
            blas_threads=1, gpu_used=False), serialization_score_check=serialization,
        per_user_sha256=hashlib.sha256((out/'per_user.npz').read_bytes()).hexdigest(),
        environment=environment(), scope='Cold-cache deterministic algorithm; '
        'timing repeats are not independent accuracy seeds.')
    write_json(out/'result.json', result)
    print(json.dumps({'dataset':args.dataset,'variant':args.variant,'split':args.split,
        'depth':args.depth,'metrics':official,'resources':result['resources']}), flush=True)

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['beauty','sports','toys'], required=True)
    p.add_argument('--variant', default='full')
    p.add_argument('--split', choices=['test','valid'], default='test')
    p.add_argument('--depth', type=int, default=1)
    p.add_argument('--strength', type=float)
    p.add_argument('--fanout', type=int)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, default=ROOT/'data')
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--save-model', action='store_true')
    run(p.parse_args())
