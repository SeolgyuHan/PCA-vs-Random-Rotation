#!/usr/bin/env python3
""" gate: the large-dataset pipeline (pass A -> freeze -> pass B -> score) run on audio must reproduce the 1M protocol's
frozen codecs and results ($PCARR_FROZEN/codec_audio__*.npz, fullscan/audio__*.csv, score_error/audio.csv,
decomp/audio.csv). Two inputs: 'val_audio_prep' = $PCARR_PREPARED/audio-db.npy written as.fvecs with mu = 0, W = I (checks
pca, rot); 'val_audio_raw' = the raw audio.fvecs with the orig codec's mu_raw, W = I (checks orig). Output results/validation/."""
import os, sys, csv, json
import numpy as np
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'); sys.path.insert(0, os.path.abspath(ROOT))
from pcafair import bigsets as B  # noqa: E402
B.ensure_dirs()
from pcafair.config import RESULTS, FROZEN, PREPARED, RAW as RAW_DIR  # noqa: E402
PREP = str(PREPARED); RAW = f'{RAW_DIR}/audio'; L2 = str(RESULTS)
OV = f'{B.OUT}/validation'; os.makedirs(OV, exist_ok=True)


def write_fvecs(path, A):
    A = np.ascontiguousarray(A, np.float32); n, d = A.shape; out = np.empty((n, d + 1), np.float32); out[:, 0] = np.frombuffer(np.int32(d).tobytes(), np.float32)[0]; out[:, 1:] = A
    out.tofile(path)


def setup():
    db = np.load(f'{PREP}/audio-db.npy', mmap_mode='r'); N, d = db.shape; q = np.load(f'{PREP}/audio-query.npy'); gt = np.load(f'{PREP}/audio-gt-l2.npy'); nq = gt.shape[0]
    write_fvecs(f'{B.WORK}/val_audio_prep-base.fvecs', np.asarray(db)); write_fvecs(f'{B.WORK}/val_audio_prep-query.fvecs', q[:nq])
    np.savez(f'{B.WORK}/val_audio_prep-stats.npz', N=N, d=d, mean=np.zeros(d), W=np.eye(d, dtype=np.float32), eig=np.ones(d))
    B.SETS['val_audio_prep'] = dict(base=f'{B.WORK}/val_audio_prep-base.fvecs', d=d, N=N, query=f'{B.WORK}/val_audio_prep-query.fvecs', nq=nq, type='audio', group='val_audio_prep', cuts=())
    co = np.load(f'{FROZEN}/codec_audio__orig.npz'); mu, act = co['mu_raw'], co['act_raw']; dr = int(np.fromfile(f'{RAW}/audio_base.fvecs', np.int32, 1)[0])
    assert act.all(), 'orig validation needs all raw dims active'
    np.savez(f'{B.WORK}/val_audio_raw-stats.npz', N=N, d=dr, mean=mu, W=np.eye(dr, dtype=np.float32), eig=np.ones(dr))
    B.SETS['val_audio_raw'] = dict(base=f'{RAW}/audio_base.fvecs', d=dr, N=N, query=f'{RAW}/audio_query.fvecs', nq=nq, type='audio', group='val_audio_raw', cuts=())
    for g in ('val_audio_prep', 'val_audio_raw'): B.GROUPS[g] = [g]
    return N, d, nq, gt


def main():
    import big_passA as PA, big_freeze as FR, big_passB as PB, big_score as SC
    N, d, nq, gt1 = setup(); rep = dict(N=N, d=d, nq=nq)
    if '--compare-only' not in sys.argv:
        for g in ('val_audio_prep', 'val_audio_raw'):
            PA.main(g); FR.main(g); PB.main(g)
        SC.main('val_audio_prep')
    gt = np.load(f'{B.OUT}/val_audio_prep-gt-l2.npy')[:, :10]
    rep['gt_identical_top10_sets'] = int(sum(set(gt[i].tolist()) == set(gt1[i, :10].tolist()) for i in range(nq)))
    for basis, src in (('pca', 'val_audio_prep'), ('rot', 'val_audio_prep'), ('orig', 'val_audio_raw')):
        a = np.load(f'{FROZEN}/codec_audio__{basis}.npz'); b = np.load(f'{B.FROZEN}/codec_{src}__{basis}.npz'); diff = {}
        for k in ('M', 'mn', 'mx', 'var', 'rlo', 'rhi', 'a', 'w', 'curve', 'c0', 'A_fix', 'g_fix'):
            if a[k].size == 0 and b[k].size == 0: continue
            diff[k] = float(np.max(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)))) if a[k].shape == b[k].shape else f'shape {a[k].shape} vs {b[k].shape}'
        rep[f'codec_{basis}_maxabsdiff'] = diff
        r1 = [r for r in csv.DictReader(open(f'{L2}/fullscan/audio__{basis}.csv')) if r['mode'] == 'unified' and r['B'] == '3'][0]
        r2 = [r for r in csv.DictReader(open(f'{B.OUT}/fullscan/{src}.csv')) if r['piece'] == basis][0]
        rep[f'recall_{basis}'] = dict(published=float(r1['recall_l2']), big=float(r2['recall_l2']), published_tie=float(r1['recall_l2_tie']), big_tie=float(r2['recall_l2_tie']))
    e1 = {(r['pair'], r['mode'], r['B']): r for r in csv.DictReader(open(f'{L2}/score_error/audio.csv'))}
    e2 = {(r['pair'], r['mode'], r['B']): r for r in csv.DictReader(open(f'{B.OUT}/score_error/val_audio_prep.csv'))}
    rel = lambda u, v: abs(u - v) / max(abs(u), abs(v), 1e-300) if (u or v) else 0.0
    rep['score_error_max_rel_diff'] = {k: max(rel(float(e1[x][k]), float(e2[x][k])) for x in e1) for k in ('J_meas', 'J_sep', 'J_hr', 'J_noise', 'c0')}
    d1 = {(r['pair'], r['B']): r for r in csv.DictReader(open(f'{L2}/decomp/audio.csv'))}
    d2 = {(r['pair'], r['B']): r for r in csv.DictReader(open(f'{B.OUT}/decomp/val_audio_prep.csv'))}
    rep['decomp_max_rel_diff'] = {k: max(rel(float(d1[x][k]), float(d2[x][k])) for x in d1) for k in ('J_meas', 'J_full', 'J_A2tr', 'c0_tr')}
    json.dump(rep, open(f'{OV}/validate_audio.json', 'w'), indent=1); print(json.dumps(rep, indent=1))
    B.log('validation (audio) -> results/validation/validate_audio.json: ' + json.dumps({k: v for k, v in rep.items() if k.startswith(('recall', 'gt_'))}))


if __name__ == '__main__':
    main()
