
# this both trains spm tokenizers and implements the
# huggingface/tokenizers bindings

import math
import os
import pathlib
import re
import sys
import typing

import datasets
import packaging.version
import sentencepiece as spm
import sentencepiece.sentencepiece_model_pb2 as model_pb2
import transformers

class SentencePieceTokenizer(transformers.tokenization_utils.PreTrainedTokenizer):
    """
    Construct a SentencePiece tokenizer using `SentencePiece <https://github.com/google/sentencepiece>`__ .

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (:obj:`str`):
            `SentencePiece <https://github.com/google/sentencepiece>`__ file (generally has a `.model` extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        bos_token (:obj:`str`, `optional`, defaults to :obj:`"<s>"`):
            A special token representing the beginning of a sentence.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            A special token representing the end of a sentence.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            A special token representing an out-of-vocabulary token.
        sep_token (:obj:`str`, `optional`):
            A special token separating two different sentences in the same input (used by BERT for instance).
        pad_token (:obj:`str`, `optional`):
            A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
            attention mechanisms or loss computation.
        cls_token (:obj:`str`, `optional`):
            A special token representing the class of the input (used by BERT for instance).
        mask_token (:obj:`str`, `optional`):
            A special token representing a masked token (used by masked-language modeling pretraining objectives, like
            BERT).
        additional_special_tokens (tuple or list of :obj:`str`, `optional`):
            A tuple or a list of additional special tokens.
    """

    vocab_files_names = {"vocab_file": "spiece.model"}
    pretrained_vocab_files_map = {"vocab_file": {}}
    max_model_input_sizes = {}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        cls_token=None,
        sep_token=None,
        mask_token=None,
        additional_special_tokens=[],
        model_max_length=1024**2,
        **kwargs
    ):
        # TODO: sentencepiece_processor.cc defines
        # `const char kSpaceSymbol[] = "\xe2\x96\x81";`
        # and we probably need to account for this somewhere
        buf = open(vocab_file, 'rb').read() if vocab_file is not None else kwargs.get('model_proto')
        model_proto = model_pb2.ModelProto()
        model_proto.ParseFromString(buf)

        ptypes = model_pb2.ModelProto.SentencePiece.Type
        pieces = { p.piece: p for p in model_proto.pieces }
        # do a quick audit of special tokens.  We can add new specials
        # and we can ignore existing specials mostly.
        # buyer beware.  If your data prep didn't use some kind of
        # escaping or encoding rules on input text then there might not
        # be a safe string that can be used to add a new special token.
        fail = False
        sync = False
        for tok, ptype, name in (
            # these are typically id 0, 1, & 2 in sentencepiece
            (unk_token, ptypes.UNKNOWN, 'unk_token'),
            (bos_token, ptypes.CONTROL, 'bos_token'),
            (eos_token, ptypes.CONTROL, 'eos_token'),
            # these may or may not have been set in SentencePieceTrainer
            (pad_token, ptypes.USER_DEFINED, 'pad_token'),
            (cls_token, ptypes.USER_DEFINED, 'cls_token'),
            (sep_token, ptypes.USER_DEFINED, 'sep_token'),
            (mask_token, ptypes.USER_DEFINED, 'mask_token'),
        ) + tuple(
            (tok, ptypes.USER_DEFINED, None) for tok in additional_special_tokens
        ):
            if tok is None:
                continue
            if tok not in pieces:
                if ptype == ptypes.USER_DEFINED:
                    model_proto.pieces.add(piece=tok, score=0.0,
                        type=ptypes.USER_DEFINED)
                    sync = True
                else:
                    raise RuntimeWarning(f'{name} "{tok}" not found in model')
                    fail = True
                continue
            if pieces[tok].type != ptype and not(
                ptype == ptypes.USER_DEFINED and
                pieces[tok].type == ptypes.CONTROL
            ):
                if ptype == ptypes.CONTROL:
                    raise RuntimeWarning(f'{name} "{tok}" is not a control token')
                elif ptype == ptypes.UNKNOWN:
                    raise RuntimeWarning(f'"{tok}" is not the unknown token')
                elif ptype == ptypes.USER_DEFINED:
                    msg = f'"{tok}" is not a user defined token'
                    if name is not None:
                        msg = f'{name} {msg}'
                    raise RuntimeWarning(msg)
                fail = True
        if fail:
            raise RuntimeError('special tokens not compatible with sentencepiece model')
        if sync:
            buf = model_proto.SerializeToString()
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            cls_token=cls_token,
            sep_token=sep_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            model_max_length=model_max_length,
            **kwargs,
        )
        self.model_proto = buf
        self.sp_model = spm.SentencePieceProcessor(model_proto=buf)

    def get_name(self):
        model_proto = model_pb2.ModelProto()
        model_proto.ParseFromString(self.model_proto)
        return os.path.split(model_proto.trainer_spec.model_prefix)[1]

    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size()

    def get_vocab(self) -> typing.Dict[str, int]:
        return {self.convert_ids_to_tokens(i): i
            for i in range(self.vocab_size)}

    def tokenize(self, text, **kwds):
        # since we're loading special tokens into sentencepiece, we can
        # directly encode here instead of in ._tokenize(), avoiding the
        # ltrim/rtrim whitespace insertion issues discussed in
        # huggingface/transformers#12308.  However this ends up not
        # honoring bos/eos settings, and *that* needs some care
        # TODO: test/debug bos/eos/cls/sep handling
        text, kwargs = self.prepare_for_tokenization(text, **kwds)
        if kwds:
            raise RuntimeWarning(f"Keyword arguments {kwds} not recognized.")
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index < self.vocab_size:
            return self.sp_model.id_to_piece(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        return self.sp_model.decode(tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: typing.Optional[str] = None) -> typing.Tuple[str]:
        if not os.path.isdir(save_directory):
            raise RuntimeWarning(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(save_directory, 
            (filename_prefix + "-" if filename_prefix else "") +
            self.vocab_files_names["vocab_file"]
        )
        open(out_vocab_file, 'wb').write(self.model_proto)
        return (out_vocab_file,)

class TrimmedSentencePieceTokenizer(SentencePieceTokenizer):
    ''' SentencePiece tokenizer max_length hint optimization.

        We are looking at some very long inputs and sending them to
        models with very short sequence length limits.  Thus encoding
        the full text and then dropping excess tokens was expensive and
        wasteful.

        So instead of `encode(text)[:max_length]`, this will try to guess
        some resonable upper limit on the input string needed for a
        `encode(text[:trim])[:max_length]` where `trim` is calculated
        using typical word length estimated from the sentencepiece
        vocabulary.

        This needs approach needs a more formal analysis eventually, but
        for now there are bigger weeds to pull in this garden.
    '''
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        s = self.sp_model
        piece_weights = (len(s.id_to_piece(i)) * math.exp(s.get_score(i))
            for i in range(s.get_piece_size()))
        # specials have score = 0, which would result in weights >= 1
        # and must be excluded
        self.avg_tok_len = sum(i for i in piece_weights if i < 1)

        self.avg_tok_len *= 1.5 # TODO: formalize this value!!
        # 1.1335 was measured as just large enough to not prematurely
        # truncate input 95% of the time in our html vocabularies on
        # the CC-MAIN-2017-13 commoncrawl sample.
        self.seq_len = None

    def tokenize(self, text, **kwds):
        # TODO: fix this disgusting hack to recover `max_length`
        seq_len = None
        level = 0
        while True:
            level += 1
            frame = sys._getframe(level)
            prnt = frame.f_locals
            if 'max_length' in prnt:
                seq_len = prnt['max_length']
                break
        if seq_len is not None:
            trim = math.ceil(seq_len * self.avg_tok_len)
            text = text[:trim]
        #else:
        #    print(RuntimeWarning('max_length not set, may be slow'))
        return super().tokenize(text, **kwds)

# make these available to AutoTokenizer.from_pretrained()
if packaging.version.parse(transformers.__version__) < \
        packaging.version.parse('4.10.0'): # Auto API refactor
    transformers.models.auto.tokenization_auto.NO_CONFIG_TOKENIZER += [
        SentencePieceTokenizer,
        TrimmedSentencePieceTokenizer,
    ]
else:
    transformers.models.auto.tokenization_auto.TOKENIZER_MAPPING_NAMES.update((
        ("mlem", ('SentencePieceTokenizer', None)),
        ("mlem", ('TrimmedSentencePieceTokenizer', None)),
    ))
    sys.modules['transformers.models.mlem'] = sys.modules[__name__]

MAX_SENTENCE_LEN = 32 * 1024
MAX_TOKEN_LEN = 64
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

def train(vocab):
    rx = r'\bcc-html([0-9]+)K(,su=f)?(,sn=f)?(,sw=f)?(,ws=t)?(,wo=t)?(,sd=t)?$'
    m = re.search(rx, vocab)
    if not m:
        raise RuntimeError(f'"{vocab}" must match {rx}')
    kwds = {}
    size = int(m[1])
    if m[2]: kwds['split_by_unicode_script'] = False
    if m[3]: kwds['split_by_number'] = False
    if m[4]: kwds['split_by_whitespace'] = False
    if m[5]: kwds['treat_whitespace_as_suffix'] = True
    if m[6]: kwds['allow_whitespace_only_pieces'] = True
    if m[7]: kwds['split_digits'] = True
    # TODO: use common path for getting at this sequence
    ds = datasets.load_dataset('parquet',
        data_files='cc-html.parquet',
        columns=('html',),
        split='train',
    ).map(
        lambda v: { 'text': btrim(v['html'], MAX_SENTENCE_LEN) },
        remove_columns=('html',),
    ).filter(lambda v: kUNKStr not in v['text'])
    pathlib.Path(os.path.split(vocab)[0]).mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(ds['text']),
        model_prefix=vocab,
        vocab_size=size<<10,
        max_sentence_length=MAX_SENTENCE_LEN,
        max_sentencepiece_length=MAX_TOKEN_LEN,
        num_threads=len(os.sched_getaffinity(0)),
        control_symbols=('<pad>'),
        user_defined_symbols=('<mask>'),
        **kwds,
    )

def get(vocab_name):
    ''' ask for a vocabulary by name, get a path useable in
        transformers.AutoTokenizer.from_pretrained() '''
    dst = f'model/spm/{vocab_name}'
    model = f'{dst}.model'
    if not os.path.exists(model):
        train(dst)
    if not os.path.exists(dst):
        TrimmedSentencePieceTokenizer(model,
            mask_token='<mask>',
            pad_token='<pad>',
        ).save_pretrained(dst)
    return dst
