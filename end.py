#!/usr/bin/env python3

''' This is an attempt at an end-to-end run of
     * vocab generation (on cc-html)
     * pretraining (also on cc-html)
     * finetuning (on formcats_wide)
'''

import argparse
import collections
import concurrent.futures
import glob
import importlib
import math
import os
import shutil
import signal
import subprocess
import sys

import torch
import transformers

import mytok # TODO: huggingface/transformers#10256 could obviate this

def configure(args, dst):
        # Mostly what this does is wire in our own tokenizer and scale
        # down models to trainable sizes on our available resources
        tok = mytok.TrimmedSentencePieceTokenizer(f'{args.vocab}.model',
            mask_token='<mask>',
            pad_token='<pad>')
        cfg = transformers.CONFIG_MAPPING[args.arch](
            vocab_size=tok.vocab_size,
            hidden_size=args.hidden,
            num_hidden_layers=args.layers,
            num_attention_heads=args.hidden // 64,
            intermediate_size=args.hidden * 4,
            tokenizer_class=(
                tok.slow_tokenizer_class if tok.is_fast else tok.__class__
            ).__name__,
            mask_token=tok.mask_token,
            max_model_length=args.max_seq_length,
        )
        cfg.save_pretrained(dst)
        tok.save_pretrained(dst)

def batch_size_autoscale(code, batch_size=60):
    ''' Try to detect application crashes due to CUDA/CPU OOMs and
        rescale batch size.  An antiprime batch_size gives best results.
        Inspired by PyTorchLightning/pytorch-lightning#1638
    '''
    cores = max(torch.cuda.device_count(), 1)
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
                    "DefaultCPUAllocator: can't allocate memory" in err.args[0]):
                continue # likely GPU memory fail, retry
            raise
    raise RuntimeError("unable to find a workable batch_size")

def train_direct(args):
    # why not subprocess.run()?  because we want to recover the error
    # object on failure, and also because we want libraries loaded in
    # this script to be available to
    # transformers.Auto*.from_pretrained()
    module = os.path.splitext(args[0])[0]
    sys.argv = list(args)
    return importlib.import_module(module).main()

def train_indirect(args):
    try:
        return (train_direct(args), None)
    except BaseException as e:
        return (None, e)

def train(*args):
    # GPU case needs to be run in-process
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return train_direct(args)
    # CPU case needs to be run in a child, due to OOM killer
    with concurrent.futures.ProcessPoolExecutor() as exe:
        res, err = exe.submit(train_indirect, args).result()
    if err is not None:
        raise err
    return res

def pretrain(src, dst):
    cfg = transformers.AutoConfig.from_pretrained(src)
    batch_size_autoscale(lambda batch_size, gradient_accumulation_steps:
        train('run_mlm.py',
            '--preprocessing_num_workers', str(len(os.sched_getaffinity(0))),
            '--seed=54321',
            '--dataset_name=pq',
            '--dataset_config_name=cc-html',
            '--do_train',
            '--do_eval',
            '--per_device_train_batch_size', str(batch_size),
            '--per_device_eval_batch_size', str(batch_size),
            '--gradient_accumulation_steps', str(gradient_accumulation_steps),
            '--max_seq_length', str(cfg.max_model_length),
            '--model_type', cfg.model_type,
            '--config_name', src,
            '--tokenizer_name', src,
            '--output_dir', dst,
        ))

def finetune(src, dst, task):
    cfg = transformers.AutoConfig.from_pretrained(src)
    batch_size_autoscale(lambda batch_size, gradient_accumulation_steps:
        train('run_glue.py',
            '--model_name_or_path', src,
            '--seed=54321',
            '--task_name', task,
            '--do_train',
            '--do_eval',
            '--per_device_train_batch_size', str(batch_size),
            '--per_device_eval_batch_size', str(batch_size),
            '--gradient_accumulation_steps', str(gradient_accumulation_steps),
            '--evaluation_strategy=steps',
            '--output_dir', dst,
            '--max_seq_length', str(cfg.max_model_length),
            '--learning_rate=2e-5',
            '--num_train_epochs=4',
        ))

def drop_checkpoints(path):
    for chkp in glob.glob(path + '/checkpoint-*'):
        shutil.rmtree(chkp)

VOCABS = [ os.path.splitext(_)[0] for _ in glob.glob('*.model') ]
MODELS = [ cf.model_type for cf in
    transformers.MODEL_FOR_MASKED_LM_MAPPING.keys() ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='end to end html language modelling test',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--vocab',
        type=str,
        choices=VOCABS,
        default='html8K')
    parser.add_argument('--arch',
        type=str,
        choices=MODELS,
        default='bert')
    parser.add_argument('-L', '--layers',
        type=int,
        default=4)
    parser.add_argument('-H', '--hidden',
        type=int,
        default=256)
    parser.add_argument('-N', '--max-seq-length',
        help=('BERT and many derived models are generally trained with '
            '128 tokens, but max out at 512 tokens, some can handle '
            'much more'),
        type=int,
        default=512)
    parser.add_argument('--finetune-epochs',
        type=int,
        default=8)
    parser.add_argument('--task',
        help='sample task to dry-run finetuning',
        type=str,
        default='formcats_wide')
    args = parser.parse_args()

    mname = 'spm/{arch}/L={layers},H={hidden},N={max_seq_length}/{vocab}'.format(**vars(args))
    mname_cfg = f'{mname}/cfg'
    mname_pre = f'{mname}/pre'
    mname_fin = f'{mname}/{args.task}'
    if not os.path.exists(mname_pre + '/eval_results.json'):
        configure(args, mname_cfg) # save settings to disk for run_mlm
        pretrain(mname_cfg, mname_pre)
        drop_checkpoints(mname_pre)

    if not os.path.exists(mname_fin + '/eval_results.json'):
        finetune(mname_pre, mname_fin, args.task)
        drop_checkpoints(mname_fin)
