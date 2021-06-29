#!/usr/bin/env python3

import json
import os

import datasets
import sentencepiece as spm

MAX_SENTENCE_LEN = 32 * 1024
kUNKStr = "▅" # sentencepiece input text can not have this character

def strim(text, size):
    ''' truncate strings in the middle, but indicate how much was
        cut out '''
    if size <= 0:
        text = ''
    elif len(text) > size:
        cut = len(text) - size + 2
        cut += len(str(cut + len(str(cut))))
        sep = f'…{cut}…'
        if size >= len(sep) + 2:
            off = (size - len(sep) + 1) // 2
            text = text[:off] + sep + text[off+cut:]
        else: # sep is too big, fallback to ellipse only
            cut = len(text) - size + 1
            off = size // 2
            text = text[:off] + '…' + text[off+cut:]
    return text

def btrim(text, size):
    # The max_sentence_length is in bytes.  Why didn't ya just make
    # the buffer caller allocated, or hey, what about a std::string?
    # You chose c++, not me, buddy. 😐
    ssize = size
    while True:
        st = strim(text, ssize)
        cut = len(bytes(st, 'utf8')) - size
        if cut > 0:
            ssize -= cut
        else:
            return st

ds = datasets.load_dataset('pq', 'cc-html', split='train')
ds = ds.map(
    lambda v: { 'text': btrim(v['html'], MAX_SENTENCE_LEN) },
    remove_columns=list(ds.features.keys()),
)
ds = ds.filter(lambda v: kUNKStr not in v['text'])

def run(size):
    vocab_size = size * 1024
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(ds['text']),
        model_prefix=f'html{size}K',
        vocab_size=vocab_size,
        max_sentence_length=MAX_SENTENCE_LEN,
        max_sentencepiece_length=64,
        num_threads=len(os.sched_getaffinity(0)),
        #split_by_unicode_script=False,
        split_by_number=False,
        #split_by_whitespace=False,
        treat_whitespace_as_suffix=True,
    )
    if False:
        # this will get the model into a format that can be read by 
        # transformers.PreTrainedTokenizerFast(tokenizer_file='file.json')
        # but we don't need that right now since 
        # transformers.models.reformer.ReformerTokenizerFast('file.model')
        # can load the spm file directly
        s = spm.SentencePieceProcessor(model_file=f'html{size}K.model')
        #s = spm.SentencePieceProcessor(model_proto=model.getvalue())
        open(f'html{size}K.json', 'wt').write(json.dumps({
            'version': '1.0',
            'truncation': None,
            'padding': None,
            'added_tokens': [],
            'normalizer': None,
            'pre_tokenizer': None,
            'post_processor': None,
            'decoder': None,
            'model': {
                'type': 'Unigram',
                'pad_id': -1,
                'unk_id': 0,
                'bos_id': 1,
                'eos_id': 2,
                'vocab': [ [ s.id_to_piece(i), s.get_score(i) ]
                    for i in range(s.vocab_size()) ],
            },
        }))

run(32)
run(16)
run(8)
run(4)
run(2)
