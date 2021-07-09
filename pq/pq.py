# coding=utf-8

from dataclasses import dataclass
from typing import Optional

import datasets
import pyarrow.parquet
import numpy

##### DATASET

VERSION = '0.0.1'

@dataclass
class PqConfig(datasets.BuilderConfig):
    """BuilderConfig for Parquet."""
    def __post_init__(self):
        # this lets `load_dataset('pq', 'data')` work shorthand for
        # `load_dataset('pq', data_files='data.parquet')`
        if self.data_files is None:
            self.data_files = self.name + '.parquet'

class PqDataset(datasets.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = PqConfig

    def _info(self):
        meta = pyarrow.parquet.read_metadata(self.config.data_files)
        return datasets.DatasetInfo(
            features=datasets.Features.from_arrow_schema(
                meta.schema.to_arrow_schema()),
        )
    def _split_generators(self, dl_manager):
        return [ datasets.SplitGenerator(
            name=datasets.Split.TRAIN,
            gen_kwargs={'file': self.config.data_files},
        ) ]
    def _generate_tables(self, file):
        yield 0, pyarrow.parquet.read_table(file)

##### METRIC

def softmax(x, axis=None):
    e_x = numpy.exp(x - numpy.max(x, axis=axis, keepdims=True))
    return e_x / numpy.sum(e_x, axis=axis, keepdims=True)

def acc(cm):
    '''returns Accuracy given N×N Confusion Matrix'''
    return cm.diagonal().sum() * cm.sum() ** -1

def mcc(cm):
    '''returns Matthews correlation coefficient given N×N Confusion Matrix'''
    n = cm.sum()
    x = cm.sum(axis=-2)
    y = cm.sum(axis=-1)
    cov_xx = numpy.sum(x * (n - x))
    cov_yy = numpy.sum(y * (n - y))
    i = cm.diagonal()
    cov_xy = numpy.sum(i * n - x * y)
    return cov_xy * (cov_xx * cov_yy) ** -0.5

@datasets.utils.file_utils.add_start_docstrings('''
    (Accuracy,MCC,Rpb,CM)
''', '''
    (Accuracy,MCC,Rpb,CM)
''')
class PqMetric(datasets.Metric):
    """(Accuracy,MCC,Rpb,CM)"""

    def _info(self):
        # TODO: Specifies the datasets.MetricInfo object
        return datasets.MetricInfo(
            # This is the description that will appear on the metrics page.
            description='(Accuracy,MCC,Rpb,CM)',
            citation=None,
            #inputs_description='(Accuracy,MCC,Rpb)',
            # This defines the format of each prediction and reference
            features=datasets.Features({
                'predictions': datasets.Sequence(datasets.Value('double')),
                'references': datasets.Value('int32'),
            }),
            ## Homepage of the metric for documentation
            #homepage=None,
            ## Additional links to the codebase or references
            #codebase_urls=None,
            #reference_urls=None,
        )

    def _compute(self, predictions, references, axis=-1):
        preds = numpy.array(predictions)
        N = preds.shape[axis]
        I = numpy.eye(N) # cheap trick to one_hot encode
        cm = numpy.matmul(I[references].T, I[numpy.argmax(preds, axis=axis)])
        # Rpb is like MCC, but scores low confidence predictions lower
        cmx = numpy.matmul(I[references].T, softmax(preds, axis=axis))
        return {
            'acc': acc(cm),
            'mcc': mcc(cm),
            'rpb': mcc(cmx),
            #'cm': cm.astype('int64').tolist(),
        }

