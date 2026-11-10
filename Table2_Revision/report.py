"""CPU-only Table 2 assembly; no sampling/calibration tables are produced."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tarfile

from spec import ROOT, CONDITIONS, configurations, data, splits, jobs, key, reuse, read, write, sha, digest, signature, verify
from task import validate_result


def grade(payload):
    from grading import evaluate, gold_answer
    k, row, item, directory, evaluator_id = payload
    destination = Path(directory)/'grading'/f'{k}.json'
    identity = digest(dict(text_sha256=row['text_sha256'],response_text=row.get('response_text'),
                           audited_correct=row.get('audited_correct'),evaluator=evaluator_id,
                           report_sha256=sha(Path(__file__)),provenance=row['provenance']))
    if destination.exists():
        old = read(destination)
        if old['identity']==identity and old['status']=='ok':
            return k, old
    if 'text' not in row:
        raise ValueError('Raw text required for uniform Table2 regrading')
    else:
        result = evaluate(row,gold_answer(item))
        result['grading_origin']='current_cpu_evaluator'
        if row['provenance']['kind']=='historical_native_raw':
            result['strict_correct']=None
        if 'audited_correct' in row:
            result.update(historical_correct=row['audited_correct'],
                          historical_grading_changed=result.get('correct')!=row['audited_correct'])
    result.update(identity=identity,index=row['index'],condition=row['condition'],config_id=row['config_id'],
                  provenance=row['provenance'],text_sha256=row['text_sha256'])
    write(destination,result)
    return k,result


def csv_out(path, rows):
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def assemble(grades, records):
    expected=jobs()
    if set(grades)!=set(expected) or set(records)!=set(expected):
        raise ValueError('Incomplete or unexpected paired Table2 records')
    rows=[]
    for config,cfg in configurations().items():
        native_g=None
        for condition in CONDITIONS:
            ids=splits()['replay']
            selected=[grades[key(config,condition,i)] for i in ids]
            if len(selected)!=200 or any(r['status']!='ok' for r in selected):
                raise ValueError('Each reported cell must contain exactly 200 valid outcomes')
            n=sum(bool(r['correct']) for r in selected)
            if condition=='G': native_g=n
            strict=[r['strict_correct'] for r in selected]
            raw=[records[key(config,condition,i)] for i in ids]
            rows.append(dict(model=cfg['size'],method=cfg['method'],dtype=cfg['dtype'],condition=condition,
                             correct_count=n,n=200,count_fraction=f'{n}/200',accuracy_percent=100*n/200,
                             gain_vs_G_pp=(n-native_g)/2,
                             strict_correct_count=sum(strict) if all(x is not None for x in strict) else None,
                             historical_label_only=sum(r['grading_origin']=='audited_historical_label_only' for r in selected),
                             reused_count=sum(r['provenance']['kind']!='new_table2_generation' for r in raw),
                             baseline_source=config.replace('_cps','_seal')))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default=str(ROOT/'runs/table2'))
    p.add_argument('--cpu-workers',type=int,default=4)
    args=p.parse_args()
    verify()
    output=Path(args.output).resolve()
    protocol=signature()
    if read(output/'RUN.json')['protocol']!=protocol or read(output/'EXPECTED.json')!=jobs():
        raise ValueError('Run manifest/protocol mismatch')
    records=reuse()
    for k,j in jobs().items():
        if k not in records:
            r=read(output/'results'/f'{k}.json.gz')
            validate_result(r,j,protocol)
            records[k]=r
    # Never fall back to a smaller denominator if a worker or evaluator failed.
    from grading import evaluator_identity
    identity=evaluator_identity()
    evaluator_id=digest(identity)
    write(output/'EVALUATOR.json',dict(id=evaluator_id,details=identity))
    items=data()
    tasks=[(k,r,items[r['index']],str(output),evaluator_id) for k,r in records.items()]
    graded={}
    with ProcessPoolExecutor(args.cpu_workers,mp_context=mp.get_context('fork')) as pool:
        for k,r in pool.map(grade,tasks,chunksize=1): graded[k]=r
    errors={k:r for k,r in graded.items() if r['status']!='ok'}
    write(output/'grading_errors.json',errors)
    write(output/'historical_grading_changes.json',{k:r for k,r in graded.items() if r.get('historical_grading_changed')})
    if errors:
        raise RuntimeError(f'{len(errors)} CPU grading errors/disagreements; no final table. Generation files retained, no GPU rerun needed.')
    rows=assemble(graded,records)
    dest=output/'summary'
    dest.mkdir(exist_ok=True)
    csv_out(dest/'table2.csv',rows)
    csv_out(dest/'per_item.csv',[dict(key=k,config_id=r['config_id'],condition=r['condition'],index=r['index'],
                  correct=graded[k]['correct'],strict_correct=graded[k]['strict_correct'],
                  source_kind=r['provenance']['kind'],text_sha256=r['text_sha256']) for k,r in records.items()])
    write(dest/'table2.json',dict(n=200,split_indices=sorted(splits()['replay']),table=rows,
          scope='Diagnostic deep200, not full MATH500; deterministic, no seed mean/SD.',
          primary='Uniform CPU regrading of all reused and new responses; historical labels are diagnostic only.',
          limitations=['CPS7 coefficient historically selected on MATH500; not held-out tuning evidence.',
                      'Historical native SEAL7/CPS15 responses are trimmed; untrimmed sensitivity unavailable, primary text grading uniform.',
                      'Transferred partial source RUN.json absent: immutable row provenance plus runtime regression gate retained.']))
    files=[p for p in output.rglob('*') if p.is_file() and p.name!='MANIFEST.json' and not p.name.startswith('.') and '.tmp.' not in p.name]
    write(output/'MANIFEST.json',{str(f.relative_to(output)):sha(f) for f in files})
    archive=output.parent/'table2_results.tar.gz'
    tmp=archive.with_suffix('.tmp')
    with tarfile.open(tmp,'w:gz') as tar:
        for f in files+[output/'MANIFEST.json']:
            tar.add(f,arcname=str(f.relative_to(output)),recursive=False)
        for name in ('reuse.json.gz','checksums.json','reuse_scope.json','cps7_provenance.json','splits/math.json'):
            tar.add(ROOT/'assets'/name,arcname='frozen_inputs/'+name,recursive=False)
    os.replace(tmp,archive)
    print(f'Complete. Return: {archive}')


if __name__=='__main__':
    main()
