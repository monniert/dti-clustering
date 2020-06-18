from collections import defaultdict, OrderedDict
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from scipy.optimize import linear_sum_assignment


class AverageMeter:
    """Compute and store the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


class AverageTensorMeter:
    """AverageMeter for tensors of size (B, *dim) over B dimension"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.count = 0

    def update(self, t):
        n = t.size(0)
        if n > 0:
            avg = t.mean(dim=0)
            self.avg = (self.count * self.avg + n * avg) / (self.count + n)
            self.count += n


class Metrics:
    def __init__(self, *names):
        self.names = list(names)
        self.curves = defaultdict(list)
        self.meters = defaultdict(AverageMeter)

    def reset(self, *names):
        if len(names) == 0:
            names = self.names
        for name in names:
            self.meters[name].reset()

    def __getitem__(self, name):
        return self.meters[name]

    def __repr__(self):
        return ', '.join(['{}={:.4f}'.format(name, self.meters[name].avg) for name in self.names])

    @property
    def avg_values(self):
        return [self.meters[name].avg for name in self.names]

    def update(self, *name_val):
        if len(name_val) == 1:
            d = name_val[0]
            assert isinstance(d, dict)
            for k, v in d.items():
                self.update(k, v)
        else:
            name, val = name_val
            if name not in self.names:
                self.names.append(name)
            if isinstance(val, (tuple, list)):
                assert len(val) == 2
                self.meters[name].update(val[0], n=val[1])
            else:
                self.meters[name].update(val)


class Scores:
    """
    Compute the following scores:
        - nmi
        - nmi diff (nmi computed with previous assignements)
        - global accuracy
        - mean accuracy
        - accuracy by class
    """
    def __init__(self, n_classes, n_clusters):
        self.n_classes = n_classes
        self.n_clusters = n_clusters
        self.n_max_labels = max(n_classes, n_clusters)
        self.names = ['nmi', 'nmi_diff', 'global_acc', 'avg_acc'] + [f'acc_cls{i}' for i in range(n_classes)]
        self.values = OrderedDict(zip(self.names, [0] * len(self.names)))
        self.score_name = 'nmi'
        self.prev_labels_pred = None
        self.reset()

    def compute(self):
        nmi = nmi_score(self.labels_true, self.labels_pred, average_method='arithmetic')
        if self.prev_labels_pred is not None:
            nmi_diff = nmi_score(self.prev_labels_pred, self.labels_pred, average_method='arithmetic')
        else:
            nmi_diff = 0

        matrix = self.compute_confusion_matrix()
        acc = np.diag(matrix).sum() / matrix.sum()
        with np.errstate(divide='ignore', invalid='ignore'):
            acc_by_class = np.diag(matrix) / matrix.sum(axis=1)
        avg_acc = np.mean(np.nan_to_num(acc_by_class))
        self.values = OrderedDict(zip(self.names, [nmi, nmi_diff, acc, avg_acc] + acc_by_class.tolist()))
        return self.values

    def __getitem__(self, k):
        return self.values[k]

    def reset(self):
        if hasattr(self, 'labels_pred'):
            self.prev_labels_pred = self.labels_pred
        self.labels_true = np.array([], dtype=np.int64)
        self.labels_pred = np.array([], dtype=np.int64)

    def update(self, labels_true, labels_pred):
        self.labels_true = np.hstack([self.labels_true, labels_true.flatten()])
        self.labels_pred = np.hstack([self.labels_pred, labels_pred.flatten()])

    def compute_confusion_matrix(self):
        # XXX 100x faster than sklearn.metrics.confusion matrix, returns matrix with GT as rows, pred as columns
        matrix = np.bincount(self.n_max_labels * self.labels_true + self.labels_pred,
                             minlength=self.n_max_labels**2).reshape(self.n_max_labels, self.n_max_labels)
        matrix = matrix[:self.n_classes, :self.n_clusters]
        if self.n_clusters == self.n_classes:
            # we find the best 1-to-1 assignent with the Hungarian algo
            best_assign_idx = linear_sum_assignment(-matrix)[1]
            matrix = matrix[:, best_assign_idx]
        else:
            # we assign each cluster to its argmax class and aggregate clusters corresponding to the same class
            # TODO improve when argmax reached for several indices
            indices = np.argmax(matrix, axis=0)
            matrix = np.vstack([matrix[:, indices == k].sum(axis=1) for k in range(self.n_classes)]).transpose()
        return matrix
