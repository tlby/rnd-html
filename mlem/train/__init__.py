
import concurrent.futures
import gc
import glob
import inspect
import math
import os
import re
import shutil
import sys
import tempfile
import weakref

import datasets
import torch
import transformers

from .. import tok
from .. import metric

def _auto_split(ds):
    sp = ds.train_test_split(1/8)
    ds_train = sp['train'].flatten_indices()
    sp = sp['test'].train_test_split(1/2)
    ds_guide = sp['train'].flatten_indices()
    ds_proof = sp['test'].flatten_indices()
    return datasets.DatasetDict({
        'train': ds_train,
        'validation': ds_guide,
        'test': ds_proof,
    })

def _tidy(path):
    shutil.rmtree(path)
    # prune any empty parent directories
    try:
        while True:
            path, _ = os.path.split(path)
            if path == '':
                break
            os.rmdir(path)
    except OSError: # directory not empty, we're done
        pass

class BatchAutoScaleTrainer(transformers.Trainer):
    ''' Try to detect application crashes due to CUDA/CPU OOMs and
        rescale batch size.  An antiprime batch_size gives best results.
        Inspired by PyTorchLightning/pytorch-lightning#1638
    '''
    def _shrink_bs(self):
        # GAS is used by both .train() and .eval() and we need to find a
        # suitable setting for both
        tbs = self.args.per_device_train_batch_size
        ebs = self.args.per_device_eval_batch_size
        gas = self.args.gradient_accumulation_steps
        for i in range(gas + 1, min(tbs, ebs) + 1):
            if tbs % i or ebs % i:
                continue
            self.args.per_device_train_batch_size = (tbs * gas) // i
            self.args.per_device_eval_batch_size = (ebs * gas) // i
            self.args.gradient_accumulation_steps = i
            return True
        return False
    def _is_oom(self, err):
        # shamelessly stolen from https://github.com/PyTorchLightning/pytorch-lightning/pull/1638/files#diff-5200c11792b86d6a07ea64820e126897aa2e3b7d3d295c92c19b141de6950afeR29-R32
        return len(err.args) == 1 and (
            "CUDA out of memory." in err.args[0]
         or "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED." in err.args[0]
         or "DefaultCPUAllocator: can't allocate memory" in err.args[0]
         or "CUDA error: CUBLAS_STATUS_ALLOC_FAILED " in err.args[0]
        )
    def _auto_scale_batch_size(self, code):
        while True:
            try:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return code()
            except RuntimeError as err:
                if self._is_oom(err) and self._shrink_bs():
                    continue
                raise
            assert(False) # bug in _shrink_bs() most likely
    def train(self, *args, **kwds):
        train = super().train
        return self._auto_scale_batch_size(
            lambda: train(*args, **kwds))
    def evaluate(self, *args, **kwds):
        evaluate = super().evaluate
        return self._auto_scale_batch_size(
            lambda: evaluate(*args, **kwds))

def get_defaults(func):
    return { k: v.default
        for k, v in inspect.signature(func).parameters.items()
        if v.default is not inspect.Parameter.empty }

def final_save(trainer):
    state = trainer.state
    args = trainer.args
    trainer.save_model()
    # fetch the final train metrics
    (train_metrics,) = filter(
        lambda log: (log['step'] == state.global_step
            and 'train_loss' in log), state.log_history)
    trainer.save_metrics("train", train_metrics)
    try:
        # fetch the final eval metrics
        (eval_metrics,) = filter(
            lambda log: (log['step'] == state.global_step
                and 'eval_loss' in log), state.log_history)
    except ValueError:
        # generate one if necessary
        trainer.evaluate()
        (eval_metrics,) = filter(
            lambda log: (log['step'] == state.global_step
                and 'eval_loss' in log), state.log_history)
    trainer.save_metrics("eval", eval_metrics)
    # and save the final trainer_state
    trainer.save_state()
    # with that done we can tidy up the training artifacts
    for ckp in trainer._sorted_checkpoints(output_dir=args.output_dir):
        _tidy(ckp)
    _tidy(args.logging_dir) # tensorboard cruft?

MODEL_CACHE_DIR = './model'
def model_path(task, vocab, model_type, model_size, max_len):
    size_names = {
        'tiny':   ( 2,  128),
        'mini':   ( 4,  256),
        'small':  ( 4,  512),
        'medium': ( 8,  512),
        'base':   (12,  768),
        'large':  (24, 1024),
    }
    if model_size in size_names:
        layers, hidden = size_names[model_size]
    else:
        m = re.match(r'^L=([0-9]+),H=([0-9]+)$', name)
        if m:
            layers, hidden = int(m[1]), int(m[2])
        else:
            raise RuntimeError(f'model_size "{model_size}" not understood')
    return os.path.join(
        MODEL_CACHE_DIR, task, model_type,
        f'L={layers},H={hidden},N={max_len}',
        vocab)

def _cfg(voc, model_type, model_size, max_len):
    cls = transformers.CONFIG_MAPPING[model_type]
    size_names = {
        'tiny':   ( 2,  128),
        'mini':   ( 4,  256),
        'small':  ( 4,  512),
        'medium': ( 8,  512),
        'base':   (12,  768),
        'large':  (24, 1024),
    }
    if model_size in size_names:
        layers, hidden = size_names[model_size]
    else:
        m = re.match(r'^L=([0-9]+),H=([0-9]+)$', model_size)
        if m:
            layers, hidden = int(m[1]), int(m[2])
        else:
            raise RuntimeError(f'model_size "{model_size}" not understood')
    mpe = get_defaults(cls.__init__).get('max_position_embeddings', 0)
    return cls(
        vocab_size=voc.vocab_size,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=hidden // 64,
        intermediate_size=hidden * 4,
        mask_token=voc.mask_token,
        max_model_length=max_len,
        # only extend this one if we must
        max_position_embeddings=max(mpe, max_len),
    )

def _lm_data(tokenizer, num_proc):
    from ..data import cc_html
    return datasets.load_dataset('parquet',
        data_files=cc_html.get(),
        columns=('html',),
    ).shuffle().map(
        lambda batch: tokenizer(batch['html']),
        batched=True,
        batch_size=256,
        num_proc=num_proc,
        remove_columns=('html',),
        desc='tokenize',
    )

def lm_trainer(dst, task_cls, **kwds):
    voc_src = tok.get(kwds.pop('vocab'))
    voc = transformers.AutoTokenizer.from_pretrained(voc_src)
    # this tokenizer for pretraining does not pin max_len, but the one
    # set on the trainer for use downstream will.
    cfg = _cfg(voc=voc, **kwds)
    args = transformers.TrainingArguments(
        output_dir=dst,
        do_train=True,
        per_device_train_batch_size=60,
        per_device_eval_batch_size=60,
        #num_train_epochs=1.0,
        max_steps=16384,
        seed=54321,

        evaluation_strategy='steps',
        eval_steps=500,
        logging_strategy='steps',
        logging_steps=500,
        save_strategy='steps',
        save_steps=500,
        save_total_limit=1,
    )
    # set the seed early
    transformers.set_seed(args.seed)
    block_size = cfg.max_model_length
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    num_proc = len(os.sched_getaffinity(0))
    data = _lm_data(voc, num_proc).map(
        group_texts,
        batched=True,
        batch_size=256,
        num_proc=num_proc,
        desc='group texts',
    )
    data = data['train'].train_test_split(4096)
    trainer = BatchAutoScaleTrainer(
        args=args,
        model=task_cls.from_config(cfg),
        # max_len should be set on the tokenizer for downstream users of
        # this pretraining, so we save a slightly modified version
        tokenizer=transformers.AutoTokenizer.from_pretrained(
            voc_src,
            model_max_length=cfg.max_model_length,
        ),
        train_dataset=data['train'],
        eval_dataset=data['test'],
        data_collator=transformers.default_data_collator,
    )
    return trainer

def pretrain(**kwds):
    dst = model_path(task='mlm', **kwds)
    if not os.path.exists(f'{dst}/config.json'):
        trainer = lm_trainer(dst, task_cls=transformers.AutoModelForMaskedLM, **kwds)
        last_checkpoint = transformers.trainer_utils.get_last_checkpoint(dst)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in trainer.model.parameters()).values())
        transformers.trainer.logger.info(
            'Training new %.3fM parameter model from scratch %s' % (
                n_params / 1000000, dst))
        trainer.train(resume_from_checkpoint=last_checkpoint)
        final_save(trainer)
    return dst

def pretrain_clm(**kwds):
    dst = model_path(task='clm', **kwds)
    if not os.path.exists(f'{dst}/config.json'):
        trainer = lm_trainer(dst, task_cls=transformers.AutoModelForCausalLM, **kwds)
        last_checkpoint = transformers.trainer_utils.get_last_checkpoint(dst)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in trainer.model.parameters()).values())
        transformers.trainer.logger.info(
            'Training new %.3fM parameter model from scratch %s' % (
                n_params / 1000000, dst))
        trainer.train(resume_from_checkpoint=last_checkpoint)
        final_save(trainer)
    return dst

def finetune_trainer(task, data, src, **kwds):
    cfg = transformers.AutoConfig.from_pretrained(src)
    voc = transformers.AutoTokenizer.from_pretrained(src,
        model_max_length=cfg.max_model_length)
    # reloading the vocab from the original source to better utilize the
    # datasets cache
    voc = transformers.AutoTokenizer.from_pretrained(tok.get(voc.get_name()),
        model_max_length=cfg.max_model_length)
    cores = max(1, transformers.TrainingArguments('.').n_gpu)
    args = dict(
        do_train=True,
        per_device_train_batch_size=60 // cores,
        per_device_eval_batch_size=60 // cores,
        num_train_epochs=4.0,
        seed=54321,

        evaluation_strategy='steps',
        eval_steps=500,
        logging_strategy='steps',
        logging_steps=500,
        save_strategy='steps',
        save_steps=500,
        save_total_limit=1,
    )
    args.update(**kwds)
    args = transformers.TrainingArguments(**args)
    transformers.set_seed(args.seed)
    num_proc = len(os.sched_getaffinity(0))
    data = data().map(
        lambda batch: voc(batch['sentence'],
            max_length=cfg.max_model_length,
            truncation=True,
            padding=True),
        batched=True,
        batch_size=256,
        num_proc=num_proc,
        remove_columns=('sentence',),
        desc='tokenize',
    )
    id2label=dict(enumerate(data['train'].features['label'].names))
    if 'validation' not in data:
        data = _auto_split(data['train'])
    met = metric.ClassificationMetric()
    # so this is a little bit awkward, but we want the caller to have
    # control of WHEN the model is brought into memory.  So instead of
    # constructing the trainer, we provide the constructor and args.
    return BatchAutoScaleTrainer, {
        'args': args,
        'model_init': lambda _=None: (
            transformers.AutoModelForSequenceClassification.from_pretrained(
                src, id2label=id2label)),
        'tokenizer': voc,
        'compute_metrics': lambda eval_pred: met.compute(
            predictions=eval_pred[0],
            references=eval_pred[1],
        ),
        'train_dataset': data['train'],
        'eval_dataset': data['validation'],
        'data_collator': transformers.default_data_collator,
    }

def finetune(task, data, **kwds):
    dst = model_path(task, **kwds)
    if not os.path.exists(f'{dst}/config.json'):
        src = pretrain(**kwds)
        tr = finetune_trainer(task, data, src, dst)
        trainer = tr[0](**tr[1])
        last_checkpoint = transformers.trainer_utils.get_last_checkpoint(dst)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in trainer.model.parameters()).values())
        transformers.trainer.logger.info(
            'Training new %.3fM parameter model %s' % (
                n_params / 1000000, dst))
        transformers.set_seed(trainer.args.seed)
        trainer.train(resume_from_checkpoint=last_checkpoint)
        final_save(trainer)
    return dst
