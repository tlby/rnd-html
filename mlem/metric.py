
import datasets
import numpy

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
    den = cov_xx * cov_yy
    return cov_xy * den ** -0.5 if den > 0 else 0


class ClassificationMetric(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description='(Accuracy,MCC,Rpb,CM)',
            citation=None,
            features=datasets.Features({
                'predictions': datasets.Sequence(datasets.Value('double')),
                'references': datasets.Value('int32'),
            }),
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
