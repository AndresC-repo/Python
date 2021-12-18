# Metrics
import torch
import torchmetrics
from dataloader.get_labels import read_labels

# Matrices - Preds and Targets:
# [class1, Class2, Class3... Classn]    Example1
# [class1, Class2, Class3... Classn]    Example2
#  ...                                    ...
# [class1, Class2, Class3... Classn]    ExampleN
#
# Every column is a class, therefore if we compare column1 from preds with column1 from Targets
# We can derive metrics for that class


class MyMetrics():
    def __init__(self, config):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__()

        self.cuda0 = torch.device('cuda:0')
        self.F1 = torchmetrics.F1(num_classes=1).to(torch.device(self.cuda0))
        self.confmat = torchmetrics.ConfusionMatrix(num_classes=2, normalize=None).to(torch.device(self.cuda0))  # noqa: E501
        self.acc = torchmetrics.Accuracy().to(torch.device(self.cuda0))
        self.prec = torchmetrics.Precision().to(torch.device(self.cuda0))
        self.recall = torchmetrics.Recall().to(torch.device(self.cuda0))
        self.labels, self.n_classes = read_labels(config['dataset'], config['detect_class'])  # noqa: E501
        self.labels.append('NO_DAM')

    def get_metrics_dic(self, preds: torch.Tensor, target: torch.Tensor, split='train'):
        self.metric = []
        recall = []
        accuracy = []
        f1 = []
        precision = []
        return_metrics = {}

        nod = torch.zeros((preds.shape[0]), device=self.cuda0)
        lab = torch.zeros((preds.shape[0]), device=self.cuda0)

        # No damage as class is derived from the absence of the other classes
        for x in range(self.n_classes):
            self.metric.append([preds[:, x], target[:, x].int()])
        for i in range(preds.shape[0]):
            nod[i] = torch.max(torch.round(preds[i, :]))
            lab[i] = torch.max(target[i, :].int())
        # we do this to make TP when there is NO damge
        nod = 1 - nod
        lab = 1 - lab

        self.metric.append([nod, lab.int()])
        # hamming distance
        hamming = [preds[:, :], target[:, :].int()]
        hm = torchmetrics.functional.hamming_distance(hamming[0], hamming[1])  # noqa: E501
        return_metrics[split + '_Hamming_Distance'] = hm

        # update metric states
        for x in range(len(self.metric)):
            accuracy.append(self.acc(self.metric[x][0], self.metric[x][1]))
            f1.append(self.F1(self.metric[x][0], self.metric[x][1]))
            recall.append(self.recall(self.metric[x][0], self.metric[x][1]))
            precision.append(self.prec(self.metric[x][0], self.metric[x][1]))

            return_metrics[split + '_' + self.labels[x] + '_acc'] = accuracy[x]
            return_metrics[split + '_' + self.labels[x] + '_f1'] = f1[x]
            return_metrics[split + '_' + self.labels[x] + '_recall'] = recall[x]  # noqa: E501
            return_metrics[split + '_' + self.labels[x] + '_precision'] = precision[x]  # noqa: E501

        return return_metrics

    def CMatrix(self, name, split='train'):
        # compute final result
        name = name + '/matrices'
        f = open(name, "a")
        for x in range(len(self.metric)):
            print('\n' + split + '_' + self.labels[x] + '_ConfusionMatrix \n',
             self.confmat(self.metric[x][0], self.metric[x][1]))  # noqa: E501
            f.write('\n' + split + '_' + self.labels[x] + '_ConfusionMatrix \n')  # noqa: E501
            f.write(str(self.confmat(self.metric[x][0], self.metric[x][1])))  # noqa: E501
        f.close()
