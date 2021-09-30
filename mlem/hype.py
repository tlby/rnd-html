
import json
import os

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
    def __init__(self, time_attr, metric, mode, conf=0.15, **kwds):
        self._time_attr = time_attr
        self._metric = metric
        self._mode = mode
        self._conf = conf
        self._stat = {}
        self.stop_times = []
        super().__init__(time_attr=time_attr, **kwds)
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

def default_space():
    space = {}
    n_trials = 1

    # architecture search params
    space['vocab'] = 'cc-html8K,sd=t'
    space['model_type'] = ray.tune.grid_search([
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
    ])
    space['model_size'] = 'tiny'
    space['max_len'] = 64

    # traditional hyperparam search
    #space['learning_rate'] = ray.tune.loguniform(
    #    5e-05 / 4,
    #    5e-05 * 4,
    #    base=4)
    #n_trials *= 2 # two tries for learning rate

    # bookkeeping junk
    space['plan_id'] = ray.tune.randint(111111, 999999)

    return n_trials, space

class FakeModel:
    def to(self, device):
        pass

def hype(name, data, space=default_space, **kwds):
    dsd = data()
    pwd = os.getcwd()
    trainer = None
    id2label = dict(enumerate(dsd['train'].features['label'].names))
    def model_init(plan):
        cwd = os.getcwd()
        os.chdir(pwd)
        if plan is None:
            # Trainer() calls model_init(None) but never does anything
            # meaningful with the result
            return FakeModel()
        src = train.pretrain(
            vocab=plan['vocab'],
            arch=plan['model_type'],
            size=plan['model_size'],
        )
        max_len = plan['max_len']
        tokenizer = transformers.AutoTokenizer.from_pretrained(src, model_max_length=max_len)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(src, id2label=id2label)
        assert(tokenizer.model_max_length == max_len)
        os.chdir(cwd)
        # now that the tokenizer has been chosen, we should reconfigure
        # the trainer a bit
        tdsd = dsd.map(lambda batch: tokenizer(batch['sentence'],
            max_length=max_len, truncation=True), batched=True,
            desc='tokenize data')
        trainer.train_dataset = tdsd['train']
        trainer.eval_dataset = tdsd['validation']
        trainer.tokenizer = tokenizer
        trainer.data_collator = transformers.data.data_collator.DataCollatorWithPadding(tokenizer)
        trainer.args.output_dir = f'{trainer.args.output_dir}/{plan["plan_id"]}'
        return model
    mcls = metric.ClassificationMetric
    m = mcls()
    epochs = 2.0
    batch_size = 60
    eval_steps = int(len(dsd['train']) * epochs / batch_size / 16)
    class TrainObserver(transformers.TrainerCallback):
        def on_train_end(self, args, state, control, **kwds):
            trainer.save_model(args.output_dir)
    trainer = transformers.Trainer(
        model_init=model_init,
        args=transformers.TrainingArguments(
            output_dir=f'{pwd}/ray/trial', # notused?
            do_train=True,
            do_eval=True,
            evaluation_strategy='steps',
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            eval_steps=eval_steps,

            disable_tqdm=True,
            seed=54321,
        ),
        compute_metrics=lambda eval_pred: m.compute(
            predictions=eval_pred[0],
            references=eval_pred[1],
        ),
        # we have to provide values for these, but will replace them in
        # model_init() once the tokenizer has been chosen
        tokenizer=False,
        data_collator=False,
        train_dataset=dsd['train'],
        eval_dataset=dsd['validation'],
        callbacks=[TrainObserver],
    )
    n_trials, config = space()
    metric_columns = [ f'eval_{k}' for k in mcls().compute(
        predictions=[[0]], references=[0]).keys() ]
    best = trainer.hyperparameter_search(
        compute_objective=lambda m: m['eval_loss'],
        direction='minimize',
        backend='ray',
        scheduler=StopOutliersScheduler(
            time_attr='epoch',
            metric='objective',
            mode='min',
        ),
        hp_space=lambda _: config,
        progress_reporter=ray.tune.CLIReporter(
            metric_columns=['epoch']+metric_columns,
        ),
        n_trials=n_trials,
        keep_checkpoints_num=1,
        local_dir=f'{pwd}/ray/trials',
    )
    # tidy up the mess
    # pass back path of winner after cleanup
    dst = f'{trainer.args.output_dir}/{best.hyperparameters["plan_id"]}'
    return dst, best.hyperparameters
