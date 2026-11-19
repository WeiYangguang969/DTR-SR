"""Fit a ranking-interval calibration using training-internal users only."""
from pathlib import Path
import sys,json,time,argparse,hashlib,datetime
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits
from experiments.redesign_calibrate import load_inner,get_spec,from_graph
from experiments.redesign_probe import ranks_dense,metrics
from dpt_sr.data import ROOT,sha256
from dpt_sr.budget_model import apply_budget
from dpt_sr.recency_calibration import kernels_for,history_experts,dense_features,prior_vector
from dpt_sr.rank_calibration import user_events,optimum,value_at
from capacity_probes.sequences import EvaluationSet
from capacity_probes.ranking import tie_ranks
from capacity_probes.counts import OffsetCache
from capacity_probes.models.pctm import count_weights
from dpt_sr.model import cache_for,fit_direction,DPTConfig

def main(a):
    tick=time.perf_counter();store,ev,verified=load_inner(a.dataset);out=ROOT/'results/redesign'/a.dataset
    if len(ev.users)>8192:
        ix=np.sort(np.random.default_rng(20260909).choice(len(ev.users),8192,replace=False))
        ev=EvaluationSet(store,ev.users[ix],ev.groups[ix],tuple(ev.targets[i] for i in ix))
    spec=get_spec(a.dataset);graph=out/'valid_graph';graph.mkdir(exist_ok=True)
    if not all((graph / name).is_file() for name in ('fit.json', 'counts.npz', 'directed.npz', 'theta.npy')):
        oc=OffsetCache(cache_for(store,ROOT/'redesign_cache'),store);kw=count_weights(spec['pctm']['counts']);oc.build(len(kw));c=oc.combine(spec['pctm']['counts'],kw).astype(float)
        d,theta,info=fit_direction(c,DPTConfig(**spec['dpt']))
        import io,zipfile
        for name,matrix in [('counts',c),('directed',d)]:
            buffer=io.BytesIO();sparse.save_npz(buffer,matrix);raw=buffer.getvalue()
            with zipfile.ZipFile(io.BytesIO(raw)) as z:assert z.testzip() is None
            tmp=graph/(name+'.tmp');tmp.write_bytes(raw);tmp.replace(graph/(name+'.npz'))
        np.save(graph/'theta.npy',theta);(graph/'fit.json').write_text(json.dumps(info,indent=2))
    c=sparse.load_npz(graph/'counts.npz');d=sparse.load_npz(graph/'directed.npz')
    theta=np.load(graph/'theta.npy');info=json.loads((graph/'fit.json').read_text());model=apply_budget(from_graph(store,spec,c,d,theta,info));del c,d
    kernels=kernels_for(spec['pctm']['kernel']);hh=history_experts(model,ev,kernels);prior=prior_vector(model,kernels);b=len(kernels)
    axes=[('popularity',-.2,.2),('path',-1.,3.)]+[(f'history_{i}',0.,1.) for i in range(1,b)]
    mask=np.asarray([int(hashlib.sha256(f'calibration:{int(u)}'.encode()).hexdigest()[:8],16)%5!=0 for u in ev.users])
    (out/'rankfit_protocol.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),dataset=a.dataset,
        split='training-internal LOO; 80/20 SHA-256 user partition',sample_users=len(ev.users),seed=20260909,
        axes=axes,coordinate_steps=1,selection='NDCG@10 on held-out internal users, including unchanged anchor',
        refit='one selected axis re-estimated on all internal sampled users',data_sha256=verified,
        optimizer='event sweep over open stable-ranking intervals; isolated crossing ties are not optimized'),indent=2))
    ties=tie_ranks(store.catalog,'sha256');initial={name:[] for name,_,_ in axes};events={name:[] for name,_,_ in axes};fixed=[]
    for start in range(0,len(ev.users),64):
        stop=min(start+64,len(ev.users));ff=dense_features(model,hh,start,stop)
        anchor=ff[0]+ff[-2]+spec['pctm']['pop_boost']*ff[-1]
        fixed.append(ranks_dense(anchor.copy(),ev,start,ties))
        for name,lo,hi in axes:
            slope=ff[-1] if name=='popularity' else ff[-2] if name=='path' else ff[int(name.split('_')[1])]-ff[0]
            for local,row in enumerate(range(start,stop)):
                s0,x,dy=user_events(anchor[local],slope[local],int(ev.targets[row][0]),store.group_items(int(ev.groups[row])),ties,lo,hi)
                initial[name].append(s0);events[name].append((x,dy))
        if start%1024==0:print('RANK-EVENTS',a.dataset,stop,len(ev.users),flush=True)
    ranks=np.concatenate(fixed);validation={};opt={};nv=int((~mask).sum());nt=int(mask.sum())
    for name,lo,hi in axes:
        ii=np.asarray(initial[name]);train_events=[e for e,m in zip(events[name],mask) if m];valid_events=[e for e,m in zip(events[name],mask) if not m]
        point,value=optimum(float(ii[mask].sum()),train_events,lo,hi)
        val=value_at(float(ii[~mask].sum()),valid_events,point)/nv
        validation[name]=val;opt[name]=dict(parameter=point,train_ndcg=value/nt,validation_ndcg=val,
            retained_events=sum(len(e[0]) for e in events[name]))
    validation['fixed']=metrics(ranks[~mask])['ndcg@10'];selected=max(validation,key=validation.get)
    final=prior.copy();parameter=0.
    if selected!='fixed':
        _,lo,hi=next(x for x in axes if x[0]==selected)
        parameter,objective=optimum(sum(initial[selected]),events[selected],lo,hi)
        if selected=='popularity':final[-1]+=parameter
        elif selected=='path':final[-2]+=parameter
        else:
            i=int(selected.split('_')[1]);final[0]=1-parameter;final[i]=parameter
    # Assert actual full-catalogue ranks at each chosen held-out coordinate,
    # independently of the event sweep (including the deterministic tie rule).
    iv=np.flatnonzero(~mask);valev=EvaluationSet(store,ev.users[iv],ev.groups[iv],tuple(ev.targets[i] for i in iv));vh=[h[iv] for h in hh]
    check=[]
    for start in range(0,nv,64):
        stop=min(start+64,nv);ff=dense_features(model,vh,start,stop);w=prior.copy()
        if selected!='fixed':
            p=opt[selected]['parameter']
            if selected=='popularity':w[-1]+=p
            elif selected=='path':w[-2]+=p
            else:i=int(selected.split('_')[1]);w[0]=1-p;w[i]=p
        score=sum(float(z)*f for z,f in zip(w,ff));check.append(ranks_dense(score,valev,start,ties))
    actual=metrics(np.concatenate(check));np.testing.assert_allclose(actual['ndcg@10'],validation[selected],atol=1e-10,rtol=0)
    result=dict(dataset=a.dataset,spec=spec,kernels=kernels,weights=final.tolist(),prior=prior.tolist(),
        selected_axis=selected,refit_parameter=parameter,validation_ndcg=validation,axis_fits=opt,validation_selected_metrics=actual,
        validation_users=nv,fit_users=nt,inner_users=len(ev.users),seconds=time.perf_counter()-tick,
        scope='Inner-validation-selected model; coefficients fitted on training-internal users.',
        algorithm_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [ROOT/'dpt_sr/budget_model.py',ROOT/'dpt_sr/recency_calibration.py',ROOT/'dpt_sr/rank_calibration.py']})
    (out/'rankfit_config.json').write_text(json.dumps(result,indent=2))
    print('RANKFIT',a.dataset,json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['beauty','sports','toys','ml1m','ml20m'],required=True)
    with threadpool_limits(limits=1):main(p.parse_args())
