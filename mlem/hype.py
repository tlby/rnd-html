
import contextlib
import itertools
import json
import os
import pathlib
import shutil
import sys
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
    def on_trial_remove(self, trial_runner, trial):
        del self._stat[trial.trial_id]
    def on_trial_result(self, trial_runner, trial, result):
        if self._metric in result and self._time_attr in result:
            if len(self._stat[trial.trial_id]):
                # the time_attr decreased?! this is bad.
                assert(result[self._time_attr] >= self._stat[trial.trial_id][0][0])
            self._stat[trial.trial_id].insert(0,
                (result[self._time_attr], result[self._metric]))
        if self._is_outlier(trial):
            del self._stat[trial.trial_id]
            return ray.tune.schedulers.trial_scheduler.TrialScheduler.STOP
        return super().on_trial_result(trial_runner, trial, result)
    def debug_string(self):
        # TODO: this is sloppy and it would be better to make the
        # mainline outlier detection algorithm inspectable
        x = self._max_x()
        if x is None:
            i, j = 0, 0
            for l in self._stat.values():
                i += 1
                j += 0 if len(l) else 1
            msg = f'waiting for {j}/{i} samples'
        else:
            Y = self._get_Y(x)
            Y = tuple(Y.values())
            from scipy.stats import t # Student's t distribution
            fit = t(*t.fit(Y))
            l = (fit.cdf(y) for y in Y)
            if self._mode == 'min':
                l = (1 - p for p in l)
            l = tuple(l)
            msg = f'@t={x} {l}'
        return f'{__class__.__name__}: {msg}'
    def save(self, checkpoint_path):
        with open(checkpoint_path, 'w') as f:
            f.write(json.dumps(self.__dict__))
    def restore(self, checkpoint_path):
        with open(checkpoint_path) as f:
            self.__dict__.update(json.loads(f.read()))
    def _is_outlier(self, trial):
        x = self._max_x()
        if x is None: # no data yet
            return False
        Y = self._get_Y(x)
        y = Y[trial.trial_id]
        Y = tuple(Y.values())
        from scipy.stats import t # Student's t distribution
        fit = t.fit(Y)
        p = t(*fit).cdf(y)
        if self._mode == 'min':
            p = 1 - p
        rv = p <= self._conf
        return rv
    def _max_x(self):
        # most recent time with points for all trials
        x = float('inf')
        for pts in self._stat.values():
            if len(pts) == 0:
                return # no data yet
            x = min(x, pts[0][0])
        return x
    def _get_Y(self, x):
        Y = {}
        for tid, pts in self._stat.items():
            p = 0
            while p + 1 < len(pts) and pts[p + 1][0] >= x:
                p += 1
            Y[tid] = pts[p][1]
        return Y

def default_grid(cost=8):
    ''' returns a list of models for triage during hp_search '''
    # there is a tradeoff between model_size and max_len we can exploit
    # during model comparisons
    model_size = [ 'medium', 'small', 'mini', 'tiny' ]
    max_len = [ 1 << _ << cost for _ in range(4) ]
    # at cost=8 this is [256, 512, 1024, 2048]
    # cost=2 is viable for cpu smoke tests.
    vocab = [
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
    ]
    model_type = [
        #'albert',        #tiny,N=2048/sphtml8K,sn=f eval_loss=5.670,
        'bert',          #tiny,N=2048/sphtml8K,sn=f eval_loss=5.677,
        #'big_bird',      #tiny,N=2048/sphtml8K,sn=f eval_loss=5.528,
        'convbert',      #tiny,N=2048/sphtml8K,sn=f eval_loss=3.203,
        #'deberta',       #tiny,N=2048/sphtml8K,sn=f eval_loss=5.681,
        #'deberta-v2',    #tiny,N=2048/sphtml8K,sn=f eval_loss=5.700,
        #'electra',       #tiny,N=2048/sphtml8K,sn=f eval_loss=5.690,
        #'layoutlm',      #tiny,N=2048/sphtml8K,sn=f eval_loss=5.673,
        'megatron-bert', #tiny,N=2048/sphtml8K,sn=f eval_loss=5.464,
        'roformer',      #tiny,N=2048/sphtml8K,sn=f eval_loss=3.759,
    ]
    for v, (s, n), t in itertools.product(vocab, zip(model_size, max_len), model_type):
        #yield train.model_path(task='mlm', vocab=v, model_type=t, model_size=s, max_len=n)
        yield train.pretrain(vocab=v, model_type=t, model_size=s, max_len=n)

def default_hp_space():
    ''' set of the traditional hp_search params (not model settings) '''
    dflt = transformers.TrainingArguments
    space = {}
    samples = 1
    if True:
        space['learning_rate'] = ray.tune.loguniform(
            dflt.learning_rate / 4,
            dflt.learning_rate * 4,
            base=4)
        samples *= 2
    if False:
        space['weight_decay'] = ray.tune.sample_from(lambda _: (
            numpy.random.gamma(0.125, 0.125)
        ))
        samples *= 2
    return samples, space

class RayTrainerObserver(transformers.TrainerCallback):
    def __init__(self):
        self.report_queue = []
        super().__init__()
    def on_train_begin(self, args, state, control, **kwds):
        # need to be careful here, seems like this can fire more than
        # once (possibly save/resume scenarios)
        tid = ray.tune.get_trial_id()
        if pathlib.Path(args.output_dir).name != tid:
            args.output_dir = str(pathlib.Path(args.output_dir, tid))
    def on_evaluate(self, args, state, control, metrics, **kwds):
        # Trainer will evaluate before checkpointing, but ray.tune
        # makes decisions from .report() so we should defer until
        # after the checkpoint save.
        self.report_queue.append(dict(metrics, step=state.global_step))
    def on_save(self, args, state, control, **kwds):
        # ray.tune wants to manage the checkpoints
        # it's easier to move an existing one than intercept the save op
        # and get it written to ray's checkpoint dir
        src = transformers.trainer_utils.get_last_checkpoint(args.output_dir)
        with ray.tune.checkpoint_dir(step=state.global_step) as dst:
            for ent in os.listdir(src):
                shutil.move(os.path.join(src, ent), os.path.join(dst, ent))
        os.rmdir(src)
        os.rmdir(args.output_dir)
        # now we can make the pending .report() calls
        while len(self.report_queue):
            ray.tune.report(**self.report_queue.pop(0))

def search(task, data, output_dir,
    grid=default_grid,
    space=default_hp_space,
    batch_size=60,
    resume=False,
    check_strategy=None,
    check_steps=500,
    num_checks=10,
):
    ctrl_args = dict(disable_tqdm=True)
    if check_strategy is not None:
        ctrl_args.update(
            evaluation_strategy=check_strategy,
            logging_strategy=check_strategy,
            save_strategy=check_strategy)
        if check_strategy == 'steps':
            ctrl_args.update(
                eval_steps=check_steps,
                logging_steps=check_steps,
                save_steps=check_steps,
                max_steps=num_checks * check_steps)
        elif check_strategy == 'epoch':
            ctrl_args.update(num_train_epochs=num_checks)
        else:
            raise RuntimeError(f'Invalid check_stategy "{check_stategy}"')

    output_dir = os.path.abspath(output_dir)
    models = {}
    for src in grid():
        key = str(pathlib.Path(src).relative_to(
            pathlib.Path(train.MODEL_CACHE_DIR)))
        models[key] = train.finetune_trainer(task, data,
            os.path.abspath(src), output_dir=output_dir, **ctrl_args)
    def trainable(trial, checkpoint_dir):
        tr = models[trial.pop('model')]
        args = tr[1]['args']
        cores = max(1, transformers.TrainingArguments('.').n_gpu)
        args.per_device_train_batch_size = batch_size // cores
        args.per_device_eval_batch_size = batch_size // cores
        # the rest of the trial settings go to training args
        for k, v in trial.items():
            setattr(args, k, v)
        # now build the trainer
        trainer = tr[0](**tr[1])
        trainer.add_callback(RayTrainerObserver())
        transformers.set_seed(trainer.args.seed)
        trainer.train(resume_from_checkpoint=checkpoint_dir)
        train.final_save(trainer)
    samples, space = space()
    config = dict(space, model=ray.tune.grid_search(list(models.keys())))
    resources_per_trial = {'cpu': 1}
    gpus = transformers.TrainingArguments('.').n_gpu
    if gpus > 0:
        cpus = len(os.sched_getaffinity(0))
        resources_per_trial['gpu'] = 1
        resources_per_trial['cpu'] = cpus // gpus
    # TODO: this isn't necessarily the metric in use by the trainer,
    # this should be exposed.
    pr = ray.tune.CLIReporter(metric_columns=[ 'step' ] + [
        f'eval_{k}' for k in metric.ClassificationMetric().outputs()
    ])
    analysis = ray.tune.run(
        trainable,
        metric='eval_rpb',
        mode='max',
        name=task,
        config=config,
        num_samples=samples,
        resources_per_trial=resources_per_trial,
        local_dir=os.path.abspath(os.path.join(output_dir, 'trials')),
        keep_checkpoints_num=1,
        progress_reporter=pr,
        scheduler=StopOutliersScheduler(time_attr='epoch'),
        resume=resume,
    )
    return analysis.best_trial.checkpoint.value, analysis
