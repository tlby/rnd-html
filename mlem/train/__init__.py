
import concurrent.futures
import glob
import inspect
import math
import os
import re
import shutil
import sys
import tempfile

import datasets
import transformers

from .. import tok
from .. import metric
from . import run_mlm
from . import run_glue

def drop_checkpoints(path):
    for chkp in glob.glob(path + '/checkpoint-*'):
        shutil.rmtree(chkp)

def batch_size_autoscale(code, batch_size=60):
    ''' Try to detect application crashes due to CUDA/CPU OOMs and
        rescale batch size.  An antiprime batch_size gives best results.
        Inspired by PyTorchLightning/pytorch-lightning#1638
    '''
    # don't ask CUDA or fork children will not get to use CUDA
    cores = max(len(glob.glob('/proc/driver/nvidia/gpus/*')), 1)
    per_core = batch_size // cores
    if per_core * cores != batch_size:
        raise RuntimeError(
            f'batch_size ({batch_size}) must divide cores ({cores}) evenly')
    # schedule from largest to smallest batch_size, with a
    # compensating gradient_accumulation_steps value for each
    n = math.floor(math.sqrt(per_core))
    fits = []
    for i in range(n, 0, -1):
        j = per_core // i
        if j * i != per_core:
            continue
        fits.insert(0, (j, i))
        if j != i:
            fits.append((i, j))
    # try the schedules until one doesn't crash
    for i, j in fits:
        try:
            return code(batch_size=i, gradient_accumulation_steps=j)
        except concurrent.futures.process.BrokenProcessPool:
            continue # likely OOM Killer on CPU memory, retry
        except RuntimeError as err:
            # shamelessly stolen from https://github.com/PyTorchLightning/pytorch-lightning/pull/1638/files#diff-5200c11792b86d6a07ea64820e126897aa2e3b7d3d295c92c19b141de6950afeR29-R32
            if len(err.args) == 1 and (
                    "CUDA out of memory." in err.args[0] or
                    "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED." in err.args[0] or
                    # adding this error to the retry list seems to have
                    # caused a kernel panic
                    #"CUDA error: CUBLAS_STATUS_ALLOC_FAILED " in err.args[0] or
                    "DefaultCPUAllocator: can't allocate memory" in err.args[0]):
                continue # likely GPU memory fail, retry
            raise
    raise RuntimeError("unable to find a workable batch_size")

def run_direct(args):
    # why not subprocess.run here?  Because we want to recover the error
    # object on failure, plus we want to shim some alterations into
    # certain scripts before running.
    func = args[0]
    sys.argv = [ func.__module__ + '.py' ] + list(args[1:])
    return func()

def train(*args):
    # this is for crash isolation, not parallelism
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as exe:
        return exe.submit(run_direct, args).result()

def get_defaults(func):
    return { k: v.default
        for k, v in inspect.signature(func).parameters.items()
        if v.default is not inspect.Parameter.empty }

def parse_size(name):
    names = {
        'tiny': (2, 128),
        'mini': (4, 256),
        'small': (4, 512),
        'medium': (8, 512),
    }
    if name in names:
        return names[name]
    rx = r'^L=([0-9]+),H=([0-9]+)$'
    m = re.match(rx, name)
    if m:
        return (int(m[1]), int(m[2]))
    raise RuntimeError(f'size "{size}" not understood')

def config(vocab, model_type='bert', model_size='tiny', max_len=64):
    layers, hidden = parse_size(model_size)
    path = (
        'model/cfg',
        model_type,
        f'L={layers},H={hidden},N={max_len}',
        vocab.get_name(),
    )
    try:
        dst = '/'.join(path)
    except:
        raise RuntimeError(f'ugh: {path}')
    if os.path.exists(f'{dst}/config.json'):
        return dst
    cls = transformers.CONFIG_MAPPING[model_type]
    cls(
        vocab_size=vocab.vocab_size,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=hidden // 64,
        intermediate_size=hidden * 4,
        mask_token=vocab.mask_token,
        max_model_length=max_len,
        # only extend this one if we must
        max_position_embeddings=max(
            get_defaults(cls.__init__).get('max_position_embeddings', 0),
            max_len)
    ).save_pretrained(dst)
    return dst

def cc_html_loader(dataset_name, dataset_config_name, **kwds):
    from ..data import cc_html
    return datasets.load_dataset('parquet',
        data_files=cc_html.get(),
        columns=('html',),
        **kwds,
    ).rename_column('html', 'text')

def pretrain(vocab, **kwds):
    voc = transformers.AutoTokenizer.from_pretrained(tok.get(vocab))
    path = config(vocab=voc, **kwds)
    cfg = transformers.AutoConfig.from_pretrained(path)
    dst = '/'.join((
        'model/mlm', cfg.model_type,
        'L={num_hidden_layers},H={hidden_size},N={max_model_length}'.format(
            **vars(cfg)), voc.get_name(),
    ))
    if os.path.exists(f'{dst}/config.json'):
        return dst
    # quick little patch to sidestep run_mlm command line arg limitations
    run_mlm.load_dataset = cc_html_loader
    batch_size_autoscale(lambda batch_size, gradient_accumulation_steps:
        train(run_mlm.main,
            '--preprocessing_num_workers', str(len(os.sched_getaffinity(0))),
            '--seed=54321',
            '--dataset_name=noop',
            '--dataset_config_name=noop',
            '--do_train',
            '--do_eval',
            '--per_device_train_batch_size', str(batch_size),
            '--per_device_eval_batch_size', str(batch_size),
            '--gradient_accumulation_steps', str(gradient_accumulation_steps),
            '--max_seq_length', str(cfg.max_model_length),
            '--model_type', cfg.model_type,
            '--config_name', path,
            '--tokenizer_name', voc.name_or_path,
            '--output_dir', dst,
            '--num_train_epochs=1',
            '--max_train_samples', str(60 * 10240),
            '--max_eval_samples', str(60 * 640),
        ))
    drop_checkpoints(dst)
    return dst

def patch_finetune(mod, data):
    # TODO: stop hot patching things
    # huggingface/transformers publishes example training scripts.  They
    # are getting more configurable with each release.  It's now at the
    # point that I don't have to edit the script to run our finetuning
    # task, but I _do_ have to hot patch in a few changes.
    mod.task_to_keys['task'] = ('sentence', None)
    # force feed dataset into load_dataset('glue', ...)
    mod.load_dataset = data
    # no-op load_metric() because we're going to replace
    # compute_metrics() anyway
    mod.load_metric = lambda *args, **kwds: None
    met = metric.ClassificationMetric()
    def compute_metrics(batch):
        return met.compute(
            predictions=batch.predictions,
            references=batch.label_ids)
    def Trainer_shim(**kwds):
        kwds['compute_metrics'] = compute_metrics
        return transformers.Trainer(**kwds)
    mod.Trainer = Trainer_shim

def finetune(task, data, **kwds):
    path = pretrain(**kwds)
    src = transformers.AutoConfig.from_pretrained(path)
    voc = transformers.AutoTokenizer.from_pretrained(path)
    dst = '/'.join((
        'model', task, src.model_type,
        'L={num_hidden_layers},H={hidden_size},N={max_model_length}'.format(
            **vars(src)),
        voc.get_name(),
    ))
    if os.path.exists(f'{dst}/config.json'):
        return dst
    patch_finetune(run_glue, data)
    batch_size_autoscale(lambda batch_size, gradient_accumulation_steps:
        train(run_glue.main,
            '--model_name_or_path', path,
            '--seed=54321',
            '--task_name=task',
            '--do_train',
            '--do_eval',
            '--per_device_train_batch_size', str(batch_size),
            '--per_device_eval_batch_size', str(batch_size),
            '--gradient_accumulation_steps', str(gradient_accumulation_steps),
            '--evaluation_strategy=steps',
            '--output_dir', dst,
            '--max_seq_length', str(src.max_model_length),
            '--learning_rate=2e-5',
            '--num_train_epochs=4',
        ))
    drop_checkpoints(dst)
    return dst
