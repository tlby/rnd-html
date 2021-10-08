
import contextlib
import itertools
import json
import os
import shutil
import sys
import traceback
import typing

import datasets
import numpy
import ray.tune
import transformers

from . import metric
from . import tok
from . import train

class TailScheduler(ray.tune.schedulers.trial_scheduler.FIFOScheduler):
    ''' Trains a group of models in parallel.  When there are more
        trials than resources, the trials that are the furthest ahead
        will be paused to make room for those which have fallen the
        furthest behind.

        Since checkpointing is not free, the strategy is not aggressive,
        but attempts to maximize the amount of progress completed in all
        trials, so that early-stopping strategies can make the most
        educated decisions.
    '''
    def __init__(self, time_attr: str = 'time_total_s'):
        self._time_attr = time_attr
        self.pause_count = 0
        super().__init__()
    def _elapsed(self, trial):
        ma = trial.metric_analysis.get(self._time_attr)
        return ma['last'] if ma else float('-inf')
    def on_trial_result(self, trial_runner, trial, result):
        if self._time_attr in result:
            # trial.metric_analysis doesn't have this result yet
            new_t = result[self._time_attr]
            above, below = 0, 0
            for t in trial_runner.get_trials():
                if t != trial:
                    above += (t.status in ('RUNNING',) and
                        new_t < self._elapsed(t))
                    below += (t.status in ('PENDING', 'PAUSED') and
                        new_t > self._elapsed(t))
            if above < below:
                self.pause_count += 1
                return ray.tune.schedulers.trial_scheduler.TrialScheduler.PAUSE
        return super().on_trial_result(trial_runner, trial, result)
    def choose_trial_to_run(self, trial_runner):
        min_t, trial = float('inf'), None
        for t in trial_runner.get_trials():
            if(t.status in ('PENDING', 'PAUSED') and
                    trial_runner.has_resources_for_trial(t)):
                cur_t = self._elapsed(t)
                if cur_t < min_t:
                    min_t = cur_t
                    trial = t
        return trial

class StopOutliersScheduler(TailScheduler):
    def __init__(self, time_attr, metric=None, mode=None, conf=0.15, **kwds):
        self._time_attr = time_attr
        self._metric = metric
        self._mode = mode
        self._conf = conf
        self._stat = {}
        self.stop_times = []
        super().__init__(time_attr=time_attr, **kwds)
    def set_search_properties(self, metric, mode):
        if metric:
            if self._metric:
                return
            self._metric = metric
        if mode:
            if self._mode:
                return
            self._mode = mode
        return True
    def on_trial_add(self, trial_runner, trial):
        self._stat[trial.trial_id] = []
    def on_trial_error(self, trial_runner, trial):
        del self._stat[trial.trial_id]
    def on_trial_complete(self, trial_runner, trial, result):
        del self._stat[trial.trial_id]
    def on_trial_remove(self, trial_runner, trial):
        del self._stat[trial.trial_id]
    def on_trial_result(self, trial_runner, trial, result):
        if self._metric in result and self._time_attr in result:
            self._stat[trial.trial_id].insert(0,
                (result[self._time_attr], result[self._metric]))
        p = self._pval(trial)
        if p <= self._conf:
            del self._stat[trial.trial_id]
            self.stop_times.append(result[self._time_attr])
            return ray.tune.schedulers.trial_scheduler.TrialScheduler.STOP
        return super().on_trial_result(trial_runner, trial, result)
    def _max_x(self):
        # most recent time with points for all trials
        x = float('inf')
        for pts in self._stat.values():
            if len(pts) == 0:
                return # no data yet
            x = min(x, pts[0][0])
        return x
    def _cur_Y(self):
        x = self._max_x()
        if x is not None:
            Y = {}
            for tid, pts in self._stat.items():
                p = 0
                while p + 1 < len(pts) and pts[p + 1][0] >= x:
                    p += 1
                Y[tid] = pts[p][1]
            return Y
    def _pval(self, trial):
        Y = self._cur_Y()
        p = float('nan')
        if Y is not None:
            y = Y[trial.trial_id]
            from scipy.stats import t # Student's t distribution
            fit = t.fit(tuple(Y.values()))
            p = t(*fit).cdf(y)
            if self._mode == 'min':
                p = 1 - p
        return p

def default_grid():
    return {
        'vocab': [
            #'cc-html8K',                # bert/mini,N=512/sphtml8K eval_loss=5.300
            'cc-html8K,sd=t',           # bert/mini,N=512/sphtml8K eval_loss=4.408
            #'cc-html8K,sn=f',           # bert/mini,N=512/sphtml8K eval_loss=5.342
            #'cc-html8K,sn=f,sw=f',      # bert/mini,N=512/sphtml8K eval_loss=5.437
            #'cc-html8K,sn=f,ws=t',      # bert/mini,N=512/sphtml8K eval_loss=5.417
            #'cc-html8K,su=f',           # bert/mini,N=512/sphtml8K eval_loss=5.912
            #'cc-html8K,su=f,sn=f',      # bert/mini,N=512/sphtml8K eval_loss=5.912
            #'cc-html8K,su=f,sw=f',      # bert/mini,N=512/sphtml8K eval_loss=5.878
            #'cc-html8K,sw=f',           # bert/mini,N=512/sphtml8K eval_loss=5.208
            #'cc-html8K,sw=f,sd=t',      # bert/mini,N=512/sphtml8K eval_loss=4.536
            #'cc-html8K,sw=f,ws=t',      # bert/mini,N=512/sphtml8K eval_loss=5.319
            #'cc-html8K,sw=f,ws=t,sd=t', # bert/mini,N=512/sphtml8K eval_loss=4.349
            #'cc-html8K,ws=t',           # bert/mini,N=512/sphtml8K eval_loss=5.359
        ],
        'model_type': [
            #'albert',        #tiny,N=2048/sphtml8K,sn=f eval_loss=5.670,
            'bert',          #tiny,N=2048/sphtml8K,sn=f eval_loss=5.677,
            #'big_bird',      #tiny,N=2048/sphtml8K,sn=f eval_loss=5.528,
            'convbert',      #tiny,N=2048/sphtml8K,sn=f eval_loss=3.203,
            #'deberta',       #tiny,N=2048/sphtml8K,sn=f eval_loss=5.681,
            #'deberta-v2',    #tiny,N=2048/sphtml8K,sn=f eval_loss=5.700,
            #'electra',       #tiny,N=2048/sphtml8K,sn=f eval_loss=5.690,
            #'layoutlm',      #tiny,N=2048/sphtml8K,sn=f eval_loss=5.673,
            #'megatron-bert', #tiny,N=2048/sphtml8K,sn=f eval_loss=5.464,
            'roformer',      #tiny,N=2048/sphtml8K,sn=f eval_loss=3.759,
        ],
        'model_size': [
            # each item is roughly half the speed of the previous
            'tiny',
            #'mini',
            #'small',
            #'medium',
        ],
        'max_len': [
            # linear speed decay
            #64, # just for smoke tests really
            #128,
            512,
            #2048,
        ],
    }

def default_hp_space():
    dflt = transformers.TrainingArguments
    space = {}
    samples = 1
    if True:
        space['learning_rate'] = ray.tune.loguniform(
            dflt.learning_rate / 4,
            dflt.learning_rate * 4,
            base=4)
        samples *= 2
    if True:
        space['weight_decay'] = ray.tune.sample_from(lambda _: (
            numpy.random.gamma(0.125, 0.125)
        ))
        samples *= 2
    return samples, space

def build_trainer(args, model, vocab, metric, data):
    ''' Just the barebones basics here '''
    def compute_metrics(eval_pred):
        return metric.compute(
            predictions=eval_pred[0],
            references=eval_pred[1],
        )
    return transformers.Trainer(
        model=model,
        args=args,
        tokenizer=vocab,
        compute_metrics=compute_metrics,
        train_dataset=data['train'],
        eval_dataset=data['validation'],
    )

def prep_trainer_for_ray(trainer, eval_freq):
    class TrainObserver(transformers.TrainerCallback):
        def __init__(self):
            self.last_eval = 0
            super().__init__()
        def on_train_begin(self, args, state, control, **kwds):
            args.output_dir = os.path.join(
                args.output_dir, ray.tune.get_trial_id())
        def on_step_end(self, args, state, control, **kwds):
            # don't use modular arithmetic here because epoch is a float
            # and the eval_freq might not divide the step count evenly
            pos = int(eval_freq * state.epoch)
            if pos > self.last_eval or state.global_step == state.max_steps:
                self.last_eval = pos
                control.should_evaluate = True
                control.should_save = True
        def on_evaluate(self, args, state, control, metrics, **kwds):
            # ray.tune needs the eval metrics as they come in
            ray.tune.report(**metrics)
        def on_save(self, args, state, control, **kwds):
            # ray.tune wants to manage the checkpoints
            cpd = transformers.trainer.PREFIX_CHECKPOINT_DIR
            src = os.path.join(args.output_dir, f'{cpd}-{state.global_step}')
            with ray.tune.checkpoint_dir(step=state.global_step) as dst:
                for ent in os.listdir(src):
                    shutil.move(os.path.join(src, ent), os.path.join(dst, ent))
            os.rmdir(src)
            # this checkpoint should have been the only entry
            os.rmdir(args.output_dir)
    trainer.add_callback(TrainObserver())
    return trainer

def hype(name, data, output_dir,
        grid=default_grid,
        space=default_hp_space,
        epochs=1.0,
        batch_size=60,
        eval_freq=8):
    output_dir = os.path.abspath(output_dir)
    data = data()
    id2label = dict(enumerate(data['train'].features['label'].names))
    met = metric.ClassificationMetric()
    grid = grid()
    keys = grid.keys()
    trainers = {}
    models = []
    for p in itertools.product(*grid.values()):
        d = dict(zip(keys, p))
        src = train.pretrain(**d)
        m_type, m_size, m_voc = src.split('/')[-3:]
        key = '/'.join((m_type, m_size, m_voc))
        cfg = transformers.AutoConfig.from_pretrained(src)
        vocab = transformers.AutoTokenizer.from_pretrained(tok.get(m_voc),
            model_max_length=cfg.max_model_length)
        trainers[key] = prep_trainer_for_ray(build_trainer(
            args=transformers.TrainingArguments(
                output_dir=output_dir,
                seed=54321,
                disable_tqdm=True,
                num_train_epochs=epochs,
            ),
            model=transformers.AutoModelForSequenceClassification.from_pretrained(
                src, id2label=id2label),
            vocab=vocab,
            metric=met,
            data=data.map(
                lambda batch: vocab(batch['sentence'], truncation=True),
                batched=True,
                desc='tokenize',
                num_proc=len(os.sched_getaffinity(0)),
            )
        ), eval_freq)
        models.append(key)
    def trainable(trial, checkpoint_dir=None):
        trial = trial.copy()
        src = trial.pop('model')
        trainer = trainers[src]
        args = trainer.args
        # the rest of the trial settings go to training args
        for k, v in trial.items():
            #prev = getattr(args, k)
            #if prev is not None and type(prev) != type(v):
            #    raise RuntimeError(f'key={k} old={type(prev)} new={type(v)}')
            setattr(args, k, v)
        # autoscale magic might work?
        def adapt(batch_size, gradient_accumulation_steps):
            trainer.args.per_device_train_batch_size = batch_size
            trainer.args.per_device_eval_batch_size = batch_size
            trainer.args.gradient_accumulation_steps = gradient_accumulation_steps
            trainer.train(resume_from_checkpoint=checkpoint_dir)
        train.batch_size_autoscale(adapt, batch_size)
        # no need to save here because ray has a final checkpoint

    samples, space = space()
    config = dict(space, model=ray.tune.grid_search(models))
    analysis = ray.tune.run(
        trainable,
        metric='eval_loss',
        mode='min',
        name=name,
        config=config,
        num_samples=samples,
        local_dir=os.path.abspath(os.path.join(output_dir, 'trials')),
        keep_checkpoints_num=1,
        progress_reporter=ray.tune.CLIReporter(metric_columns=[ 'epoch' ] + [
            f'eval_{k}' for k in met.outputs()
        ]),
        scheduler=StopOutliersScheduler(
            time_attr='epoch',
        ),
    )
    return analysis.best_trial.checkpoint.value, analysis
