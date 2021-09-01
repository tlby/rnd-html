
import json

from . import train

DEFAULT_PLAN = {
    'arch': (
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
    ),
    'vocab': ('cc-html8K',),
    'vopt': (
        #'',                # bert/mini,N=512/sphtml8K eval_loss=5.300
        ',sd=t',           # bert/mini,N=512/sphtml8K eval_loss=4.408
        #',sn=f',           # bert/mini,N=512/sphtml8K eval_loss=5.342
        #',sn=f,sw=f',      # bert/mini,N=512/sphtml8K eval_loss=5.437
        #',sn=f,ws=t',      # bert/mini,N=512/sphtml8K eval_loss=5.417
        #',su=f',           # bert/mini,N=512/sphtml8K eval_loss=5.912
        #',su=f,sn=f',      # bert/mini,N=512/sphtml8K eval_loss=5.912
        #',su=f,sw=f',      # bert/mini,N=512/sphtml8K eval_loss=5.878
        #',sw=f',           # bert/mini,N=512/sphtml8K eval_loss=5.208
        ',sw=f,sd=t',      # bert/mini,N=512/sphtml8K eval_loss=4.536
        #',sw=f,ws=t',      # bert/mini,N=512/sphtml8K eval_loss=5.319
        ',sw=f,ws=t,sd=t', # bert/mini,N=512/sphtml8K eval_loss=4.349
        #',ws=t',           # bert/mini,N=512/sphtml8K eval_loss=5.359
    ),
    'size': ('tiny', 'mini'),
    'max_len': (512, 1024, 2048),
}

def score(task, data, opts):
    opts = opts.copy()
    opts['vocab'] += opts['vopt']
    del opts['vopt']
    path = train.finetune(task, data, **opts)
    try:
        with open(f'{path}/eval_results.json', 'rt') as f:
            res = json.loads(f.read())
        # TODO: we'd really rather be scoring based on the
        # loss/samples_per_second curve but that's trickier
        return (res['eval_loss'], path)
    #except FileNotFoundError:
    #    return (float('NaN'), path)
    finally:
        pass

def hype(name, data, plan=DEFAULT_PLAN, k=3):
    phase = 0
    # test a seed case
    seed = { key: plan[key][0] for key in plan.keys() }
    best = [ (score(name, data, seed), phase, seed) ]
    # compare variants to the best discovered so far
    for key, vals in plan.items():
        if len(plan[key]) < 2: # no choices to explore
            continue
        test = []
        phase += 1
        for prev in best[:k]:
            for val in vals[1:]: # vals[0] has already been scored
                opts = dict(prev[-1], **{ key: val })
                sc = score(name, data, opts)
                if opts:
                    test.append((sc, phase, opts))
        best = sorted(best + test, key=lambda v: v[0])
    return [ v[:-1] for v in best ]
