"""Where the experiments read and write. Override with environment variables (defaults in parentheses):

PCARR_RAW raw datasets, <name>/<name>_{base,query}.fvecs (data/raw)
PCARR_PREPARED prepared datasets, <name>-{db,query,gt-l2,eigenvalues}.npy (data/prepared)
PCARR_WORK caches and working files (work)
PCARR_OUT new result files, in the layout of results/ ($PCARR_WORK/results)
PCARR_FROZEN frozen codecs, codec_<name>__<basis>.{npz,json} ($PCARR_WORK/frozen)
PCARR_BIG_RAW raw files of the large datasets (SIFT100M/, MSMARCO100M/) ($PCARR_RAW)
PCARR_BIG_WORK streaming-pass checkpoints and statistics of the large datasets ($PCARR_WORK/bigsets)
Every experiment writes under PCARR_OUT, so a run never overwrites the published files in results/."""
import os, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get('PCARR_RAW', ROOT / 'data' / 'raw'))
PREPARED = Path(os.environ.get('PCARR_PREPARED', ROOT / 'data' / 'prepared'))
WORK = Path(os.environ.get('PCARR_WORK', ROOT / 'work'))
OUT = Path(os.environ.get('PCARR_OUT', WORK / 'results'))
FROZEN = Path(os.environ.get('PCARR_FROZEN', WORK / 'frozen'))
BIG_RAW = Path(os.environ.get('PCARR_BIG_RAW', RAW))
BIG_WORK = Path(os.environ.get('PCARR_BIG_WORK', WORK / 'bigsets'))
RESULTS = ROOT / 'results'                       # the published result files (read only)
NUMBERS = ROOT / 'numbers' / 'numbers.json'      # the numbers printed in the paper

TEN = ['youtube', 'word2vec', 'openai1536', 'msmarco', 'openai3072', 'gist', 'fashion-mnist', 'audio', 'msong', 'mnist']   # CG order
LARGE = ['sift100m', 'msmarco10m', 'msmarco113m']
# sub-folders of results/ (and of PCARR_OUT)
SUB = dict(fullscan='fullscan', rerank='rerank', score='score_error', score_b57='score_error_b57', score_orig='score_error_orig',
           decomp='decomp', decomp_orig='decomp_orig', guarantee='guarantee', calc='calc', spectra='spectra', big='bigsets')


def numbers():
    return json.load(open(NUMBERS, encoding='utf-8'))


def set_threads(n):
    """Fix the BLAS thread count before numpy is imported (the published runs used fixed counts; floating-point reductions,
    and hence the last digits of some results, depend on it)."""
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(v, str(n))
