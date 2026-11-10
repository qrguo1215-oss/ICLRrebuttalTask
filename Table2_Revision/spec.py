"""Frozen Table 2 protocol. Planning requires only the Python standard library."""
import gzip
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VERSION = "table2-deep200-v1"
CONDITIONS = ("G", "V", "token_replay", "thought_replay")


def read(path):
    with (gzip.open if str(path).endswith('.gz') else open)(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with (gzip.open if str(path).endswith('.gz') else open)(tmp, 'wt', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False)
    with tmp.open('rb') as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def configurations():
    return {f'math{s}_{m}': dict(id=f'math{s}_{m}', size=s, method=m, dataset='math',
                dtype='float16' if m == 'rebalance' else 'bfloat16',
                layer=19 if s == '15' and m == 'rebalance' else 20,
                coefficient=1.0 if m == 'cps' else -1.0, vector=f'assets/vectors/{s}_{m}.pt')
            for s in ('15', '7') for m in ('seal', 'rebalance', 'cps')}


def data():
    rows = [json.loads(s) for s in (ROOT/'assets/data/math.jsonl').read_text().splitlines() if s.strip()]
    return {int(r['index']): r for r in rows}


def splits():
    return read(ROOT/'assets/splits/math.json')


def key(config, condition, index):
    if condition == 'G':
        config = config.replace('_cps', '_seal')
    return f'{config}/{condition}/{int(index):04d}'


def jobs():
    result = {}
    for config in configurations():
        for condition in CONDITIONS:
            for i in splits()['replay']:
                k = key(config, condition, i)
                result[k] = dict(config_id=k.split('/')[0], condition=condition, index=i, seed=0)
    return result


def reuse():
    return read(ROOT/'assets/reuse.json.gz')


def signature():
    names = ['spec.py', 'engine.py', 'native.py', 'requirements.txt']
    names += [str(p.relative_to(ROOT)) for p in (ROOT/'assets').rglob('*') if p.is_file()]
    names += [str(p.relative_to(ROOT)) for p in (ROOT/'vendor').glob('*model.py')]
    return digest({'version':VERSION, 'files':{n:sha(ROOT/n) for n in sorted(names)}})


def verify():
    manifest = read(ROOT/'assets/checksums.json')
    for name, h in manifest.items():
        if sha(ROOT/name) != h:
            raise ValueError(f'asset modified: {name}')
    s = splits()
    assert len(s['replay']) == len(set(s['replay'])) == 200
    assert set(s['replay']) <= set(s['test'])
    assert not set(s['calibration']) & set(s['replay'])
    assert len(data()) == 500
    expected = jobs()
    for k, r in reuse().items():
        assert k in expected and key(r['config_id'], r['condition'], r['index']) == k
        assert r['max_new_tokens'] == 10000 and r['dtype'] == configurations()[r['config_id']]['dtype']
        assert r['provenance'] and not r.get('smoke', False)
        assert 'text' in r, 'Every formal reused record must support uniform CPU regrading'
        assert hashlib.sha256(r['text'].encode()).hexdigest() == r['text_sha256']
    return {'table_cells':24, 'logical_records':4800, 'unique_jobs':len(expected), 'reused':len(reuse()),
            'missing':len(set(expected)-set(reuse())), 'fixed_n':200}


def generation_seed(config, index, seed, boundary_sampling=False):
    # No sampling in this task; retained for the frozen inference engine interface.
    return 20260920
