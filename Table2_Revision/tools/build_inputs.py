"""Maintainer-only deterministic export. Collaborators never need to run this."""
import argparse
import json
import shutil
import sys
import tarfile
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spec import sha, write, read, configurations, key, jobs, splits


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--partial', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--cps7', type=Path, required=True)
    p.add_argument('--cps15-native', type=Path, required=True)
    p.add_argument('--seal7-archive', type=Path, required=True)
    args = p.parse_args()
    old = ROOT.parent/'Steering_Revision'
    for group, names in [('data',['math.jsonl']), ('splits',['math.json']),
                         ('vectors',['15_seal.pt','15_rebalance.pt','15_cps.pt','7_seal.pt','7_rebalance.pt'])]:
        for name in names:
            dest = ROOT/'assets'/group/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(old/'assets'/group/name, dest)
    shutil.copyfile(old/'assets/models.json', ROOT/'assets/models.json')
    shutil.copyfile(args.cps7/'localbest_transferred.pt', ROOT/'assets/vectors/7_cps.pt')
    # Only portable, relevant source facts; no credentials or deployment paths.
    write(ROOT/'assets/cps7_provenance.json', dict(vector_sha256=sha(args.cps7/'localbest_transferred.pt'),
        source_receipt_sha256=sha(args.cps7/'RESULT_RECEIPT.json'), formula='-0.6*C5 already baked into tensor',
        runtime_coefficient=1.0, layer=20, evaluation_dtype='bfloat16',
        legacy_selection_warning='Historical -0.6 coefficient was selected on MATH500; not a fresh held-out efficacy test.',
        old_predictions_not_reused='Source evaluator does not explicitly set torch_dtype or save runtime/token IDs; not certified BF16.'))
    wanted = jobs()
    accepted = {}
    for k, j in wanted.items():
        source = args.partial/'results'/j['config_id']/j['condition']/f"{j['index']:04d}_0.json.gz"
        if not source.exists():
            continue
        r = read(source)
        assert r['config_id'] == j['config_id'] and r['condition'] == j['condition'] and r['index'] == j['index']
        assert not r['smoke'] and r['max_new_tokens'] == 10000
        r['provenance'] = dict(kind='collaborator_raw', relative_source=str(source.relative_to(args.partial)),
            file_sha256=sha(source), original_run_id=r.pop('run_id'),
            limitation='Original RUN.json absent from received partial archive; row metadata and offline integrity verified. Runtime regression gate required before continuation.')
        accepted[k] = r
    # Historical label-only records are sufficient for paired outcome statistics.
    # They are never passed to the model as a replay prefix or assigned a new run ID.
    for location in ('remote',):
        extract = read(args.audit/f'{location}_raw_extract.json')
        extract_hash = sha(args.audit/f'{location}_raw_extract.json')
        for r in extract['rows']:
            src = extract['files'][r['source_file_id']]
            config = None
            for name, prefix in [('math7_seal','cross_model_method_replication_v1/priority1/anchors/seal7/')]:
                if '/experiments/runs/'+prefix in src['path']:
                    config = name
            if not config:
                continue
            k = key(config, 'V', r['index'])
            if k not in wanted:
                continue
            correct = r.get('correct', r.get('mv_eval'))
            if isinstance(correct,list):
                assert len(correct) == 1
                correct = correct[0]
            assert isinstance(correct, bool)
            row = dict(wanted[k], dtype=configurations()[config]['dtype'], max_new_tokens=10000,
                       audited_correct=correct, text_sha256=r['text_sha256'], token_count=r['token_count'],
                       cache_policy='native', hit_max=r.get('hit_max'), closed=r.get('closed',r.get('closure')),
                       provenance=dict(kind='audited_historical_outcome', extract_sha256=extract_hash,
                           source_file_sha256=src['sha256'], source_line=r.get('source_line'),
                           source_relative=src['path'].split('/experiments/runs/')[-1],
                           source_seed=r.get('seed'), grading='Historical primary evaluator label; not regraded without raw response.',
                           limitation='Complete source response/token IDs not in extract; reuse only for native outcome statistics.'))
            if k in accepted:
                assert accepted[k]['text_sha256'] == row['text_sha256']
            else:
                accepted[k] = row
    for line in args.cps15_native.read_text().splitlines():
        original = json.loads(line)
        k = key('math15_cps','V',original['index'])
        if k not in wanted:
            continue
        text = original['model_generation'][0]
        assert isinstance(text,str), list(original)
        n = original['token_count']
        accepted[k] = dict(wanted[k], dtype='bfloat16',max_new_tokens=10000,
            text=text,response_text=text,text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            audited_correct=bool(original['mv_eval']),token_count=n[0] if isinstance(n,list) else n,
            cache_policy='native',hit_max=original['hit_max'],closed=original['closure'],
            provenance=dict(kind='historical_native_raw',raw_source_sha256=sha(args.cps15_native),raw_text_available=True,
                source_seed=original['seed'],source_relative='math500_trajectory_resampling_v1/reference/localbest.jsonl',
                response_text_scope='Historical text only; strict untrimmed sensitivity is unavailable.'))
    # Recover archived raw text and verify against the independently audited shard hashes.
    with tarfile.open(args.seal7_archive) as tar:
        for member in tar.getmembers():
            if not member.name.startswith('seal7/shard') or not member.name.endswith('.jsonl'):
                continue
            assert member.isfile() and not member.issym() and not member.islnk()
            content = tar.extractfile(member).read()
            source_sha = hashlib.sha256(content).hexdigest()
            for line in content.decode().splitlines():
                original = json.loads(line)
                k = key('math7_seal','V',original['index'])
                if k not in accepted:
                    continue
                r = accepted[k]
                assert source_sha == r['provenance']['source_file_sha256']
                assert original['text_sha256'] == r['text_sha256']
                assert hashlib.sha256(original['text'].encode()).hexdigest() == r['text_sha256']
                assert original['dtype']=='bfloat16' and original['layer']==20 and original['coefficient']==-1.0
                r.update(text=original['text'],response_text=original['text'])
                r['provenance'].update(kind='historical_native_raw',raw_text_available=True,
                    source_code_sha256=original['code_sha256'],response_text_scope='Historical trimmed text; untrimmed sensitivity unavailable.',
                    limitation='Raw shard recovered and independently matched to audit hashes; source deployment not relabeled as current run.')
    assert all('text' in r for r in accepted.values()), 'Raw response missing: do not mix historical labels with new grading'
    write(ROOT/'assets/reuse.json.gz', accepted)
    # Two shortest calibration cases per actual baseline wrapper, never chosen by correctness.
    probes = {}
    for cfg in ('math15_seal','math15_rebalance','math7_seal','math7_rebalance'):
        records = [read(args.partial/'results'/cfg/'G'/f'{i:04d}_0.json.gz') for i in splits()['calibration']]
        probes[cfg] = sorted(records,key=lambda r:(r['token_count'],r['index']))[:2]
    write(ROOT/'assets/regression.json.gz', probes)
    write(ROOT/'assets/reuse_scope.json', dict(accepted=len(accepted), pending=[j for k,j in wanted.items() if k not in accepted],
        non_reuse_reasons=['Old 7B thought120 uses distinct donor-history protocol; do not equate by count or cap alone.',
                          'Old CPS thought128 is not full thought; CPS token raw results unavailable.',
                          'ReBalance7 V200: old cross-problem _prev_step_mean reset not certified equivalent; do not mix with corrected replay.',
                          '7B CPS vector accepted but old generation precision not certified.'],
        audit_report_sha256=sha(args.audit/'REPORT.zh.md')))
    files = [f for f in (ROOT/'assets').rglob('*') if f.is_file() and f.name != 'checksums.json']
    write(ROOT/'assets/checksums.json',{str(f.relative_to(ROOT)):sha(f) for f in files})
    print({'reused':len(accepted),'missing':len(wanted)-len(accepted)})


if __name__ == '__main__':
    main()
