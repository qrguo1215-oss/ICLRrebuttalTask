#!/usr/bin/env python3
"""Table 2 only: reuse audited records, execute missing deterministic jobs."""
import argparse
import ctypes
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from spec import ROOT, CONDITIONS, configurations, data, splits, jobs, key, reuse, read, write, digest, sha, signature, verify


def reject_model_overrides(path, expected):
    bad = [p.name for p in Path(path).iterdir() if p.is_file() and p.name not in expected and
           (p.suffix in ('.bin','.pt','.pth','.safetensors','.gguf') or p.name.startswith(
               ('adapter','tokenizer','added_tokens','special_tokens','vocab','merges')) or p.name.endswith('.index.json'))]
    if bad:
        raise ValueError(f'Unexpected model/tokenizer files: {bad}')


def ensure_models(model_root, config_ids):
    from huggingface_hub import snapshot_download
    manifest = read(ROOT/'assets/models.json')
    result = {}
    for size in sorted({configurations()[c]['size'] for c in config_ids}):
        info = manifest[size]
        dest = Path(model_root).resolve()/info['directory']
        if not all((dest/n).is_file() for n in info['files']):
            snapshot_download(repo_id=info['repo_id'], revision=info['revision'], local_dir=str(dest), allow_patterns=list(info['files']))
        reject_model_overrides(dest, info['files'])
        for name, entry in info['files'].items():
            if sha(dest/name) != entry['sha256']:
                raise ValueError(f'Model checksum mismatch: {name}')
        result[size] = str(dest)
    return result


def configure_runtime():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    import torch
    import transformers
    import numpy
    torch.set_num_threads(2)
    torch.manual_seed(20260920)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA with native BF16 support required')
    if not str(torch.__version__).startswith('2.3.1') or transformers.__version__ != '4.46.3':
        raise RuntimeError('Use the pinned requirements; do not mix inference versions')
    return dict(torch=str(torch.__version__), transformers=transformers.__version__, numpy=numpy.__version__,
                cuda=torch.version.cuda, gpu_name=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                attention='sdpa', tf32=False, batch_size=1)


def validate_result(row, job, protocol):
    if row.get('protocol') != protocol or row.get('smoke') or row.get('max_new_tokens') != 10000:
        raise ValueError('Result protocol/smoke/length mismatch')
    if any(row[k] != job[k] for k in ('config_id','condition','index','seed')):
        raise ValueError('Result job mismatch')
    cfg = configurations()[job['config_id']]
    expected_cache = 'clean_replay' if job['condition'].endswith('replay') else 'native' if job['condition']=='V' else 'unsteered'
    if row['dtype'] != cfg['dtype'] or row['cache_policy'] != expected_cache:
        raise ValueError('Precision or cache-policy mismatch')
    if sha_text(row['text']) != row['text_sha256']:
        raise ValueError('Text checksum mismatch')
    if row['generated_tokens_including_eos'] != len(row['token_ids']) or len(row['token_ids']) > 10000:
        raise ValueError('Token count mismatch')


def sha_text(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def pending(output, config_ids):
    accepted = reuse()
    protocol = signature()
    result = {}
    for k, job in jobs().items():
        if job['config_id'] not in config_ids or k in accepted:
            continue
        path = Path(output)/'results'/f'{k}.json.gz'
        if path.exists():
            validate_result(read(path), job, protocol)
        else:
            result[k] = job
    return result


def safety_gate(engine, cfg, records):
    source = cfg['id'].replace('_cps','_seal')
    probes = read(ROOT/'assets/regression.json.gz')[source]
    details = []
    for reference in probes:
        item = records[reference['index']]
        row = engine.generate(item, 'G')
        if row['token_ids'] != reference['token_ids']:
            raise RuntimeError('Historical baseline regression failed. Stop; preserve outputs and inspect runtime. Do not silently regenerate all references.')
        details.append(dict(index=item['index'], full_tokens_match=True, tokens=row['token_count']))
    item = records[probes[0]['index']]
    audit = engine.audit(item)
    g = engine.generate(item,'G',max_tokens=256)['token_ids']
    for condition in ('G_off','self_token','self_thought'):
        if engine.generate(item,condition,max_tokens=256)['token_ids'] != g:
            raise RuntimeError(f'Identity gate failed: {condition}')
    v = engine.generate(item,'V',max_tokens=256)['token_ids']
    if v != engine.generate(item,'V',max_tokens=256,record_shadow=False)['token_ids']:
        raise RuntimeError('Shadow-state isolation failed')
    if cfg['method']=='rebalance':
        import torch
        engine.model._prev_step_mean = torch.tensor([.123],device=engine.model.device)
        engine.model._last_maxprob = torch.tensor([.987],device=engine.model.device)
        if v != engine.generate(item,'V',max_tokens=256)['token_ids']:
            raise RuntimeError('Cross-problem reset failed')
    return dict(regression=details,audit=audit,identity=True,shadow_isolation=True)


def worker(device, cfgid, model_path, work, output, protocol, smoke, parent_pid):
    if sys.platform == 'linux':
        if ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) != 0 or os.getppid()!=parent_pid:
            raise RuntimeError('Parent-death guard failed')
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    runtime = configure_runtime()
    from native import load_model
    from engine import Engine
    cfg = configurations()[cfgid]
    model, tokenizer, vector = load_model(cfg, model_path)
    engine = Engine(model,tokenizer,vector,cfg)
    records = data()
    stamp = digest(dict(protocol=protocol,runtime=runtime,config=cfgid))
    gate_path = Path(output)/'smoke'/cfgid/f'{stamp}.json'
    if gate_path.exists():
        saved = read(gate_path)
        if saved.get('stamp')!=stamp or saved.get('protocol')!=protocol or saved.get('passed') is not True:
            raise ValueError('Invalid saved safety gate')
    else:
        result = safety_gate(engine,cfg,records)
        write(gate_path,dict(stamp=stamp,protocol=protocol,runtime=runtime,passed=True,details=result))
    if smoke:
        return
    while True:
        entry = work.get()
        if entry is None:
            return
        k, job = entry
        start = time.monotonic()
        row = engine.generate(records[job['index']],job['condition'],max_tokens=10000)
        row.update(job,protocol=protocol,smoke=False,runtime=runtime,dtype=cfg['dtype'],
                   wall_seconds=time.monotonic()-start,
                   provenance=dict(kind='new_table2_generation',vector_sha256=sha(ROOT/cfg['vector']),safety_gate=stamp))
        validate_result(row,job,protocol)
        write(Path(output)/'results'/f'{k}.json.gz',row)
        print(json.dumps(dict(done=k,tokens=row['token_count'],seconds=row['wall_seconds'])),flush=True)


def run(args):
    verify()
    configs = list(configurations()) if not args.configs else args.configs.split(',')
    if not set(configs)<=set(configurations()):
        raise ValueError('Unknown Table2 config')
    output = Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    lock = (output/'.controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol = signature()
    meta = dict(protocol=protocol,version='table2-only',n=200,max_new_tokens=10000,
                split_sha256=digest(sorted(splits()['replay'])),source_reuse_sha256=sha(ROOT/'assets/reuse.json.gz'))
    if (output/'RUN.json').exists() and read(output/'RUN.json')!=meta:
        raise ValueError('Output belongs to another scientific protocol')
    write(output/'RUN.json',meta)
    write(output/'EXPECTED.json',jobs())
    todo = pending(output,configs)
    active = configs if args.command=='smoke' else [c for c in configs if any(j['config_id']==c for j in todo.values())]
    if active:
        if not args.gpus or args.workers_per_gpu<1:
            raise ValueError('Provide --gpus 0,1,... and positive --workers-per-gpu')
        devices = args.gpus.split(',')
        if len(devices)!=len(set(devices)) or any(not x for x in devices):
            raise ValueError('GPU identifiers must be unique and nonempty')
        models = ensure_models(args.model_root,active)
        ctx = mp.get_context('spawn')
        old = reuse()
        for cfgid in active:
            work = ctx.Queue()
            items = [(k,j) for k,j in todo.items() if j['config_id']==cfgid]
            items.sort(key=lambda x:-old[key(cfgid,'G',x[1]['index'])]['token_count'])
            slots = devices if args.command=='smoke' else [d for d in devices for _ in range(args.workers_per_gpu)][:len(items)]
            for item in items:
                work.put(item)
            for _ in slots:
                work.put(None)
            processes = []
            try:
                for device in slots:
                    process = ctx.Process(target=worker,args=(device,cfgid,models[configurations()[cfgid]['size']],work,str(output),protocol,args.command=='smoke',os.getpid()))
                    process.start()
                    processes.append(process)
                next_monitor = 0.
                while any(p.is_alive() for p in processes):
                    if any(p.exitcode not in (None,0) for p in processes):
                        raise RuntimeError('Worker failed; completed outputs retained. Do not ignore the failure.')
                    if time.monotonic()>=next_monitor:
                        next_monitor=time.monotonic()+30
                        try:
                            stats=subprocess.run(['nvidia-smi','--id='+','.join(devices),'--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                            with (output/'gpu.jsonl').open('a') as f:
                                f.write(json.dumps(dict(time=time.time(),config=cfgid,stats=stats.stdout.strip()))+'\n')
                        except (OSError,subprocess.TimeoutExpired):
                            pass
                    time.sleep(1)
                if any(p.exitcode!=0 for p in processes):
                    raise RuntimeError('Worker failure')
            finally:
                for p in processes:
                    if p.is_alive(): p.terminate()
                    p.join(timeout=15)
                    if p.is_alive(): p.kill(); p.join()
                work.cancel_join_thread()
                work.close()
            if args.command!='smoke' and pending(output,[cfgid]):
                raise RuntimeError('Queue incomplete; restart same command to resume')
    if args.command=='smoke':
        print('Safety gates passed. No test-set generation started.')
    elif pending(output,list(configurations())):
        print('Assigned configs complete. Other configs remain; rerun with the remaining --configs.')
    else:
        subprocess.run([sys.executable,str(ROOT/'report.py'),'--output',str(output),'--cpu-workers',str(args.cpu_workers)],check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['plan','verify','smoke','run'])
    p.add_argument('--gpus')
    p.add_argument('--workers-per-gpu',type=int,default=1)
    p.add_argument('--configs')
    p.add_argument('--output',default=str(ROOT/'runs/table2'))
    p.add_argument('--model-root',default=str(ROOT/'models'))
    p.add_argument('--cpu-workers',type=int,default=4)
    args=p.parse_args()
    def stop(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}: completed records remain resumable')
    signal.signal(signal.SIGTERM,stop)
    if args.command in ('plan','verify'):
        info=verify()
        todo=pending(args.output,list(configurations()))
        from collections import Counter
        info.update(remaining=len(todo),by_condition=dict(Counter(j['condition'] for j in todo.values())),
                    by_config=dict(Counter(j['config_id'] for j in todo.values())))
        print(json.dumps(info,indent=2))
    else:
        run(args)


if __name__=='__main__':
    main()
