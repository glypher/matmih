"""features.py: Helper classes to evaluate models
and compare results using statistical methods
"""
__author__ = "Mihai Matei"
__license__ = "BSD"
__email__ = "mihai.matei@my.fmi.unibuc.ro"


from .model import ModelHistorySet, DataType
from .plot import PlotBuilder

import seaborn as sns
import pandas as pd
from scipy import stats
import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from prettytable import PrettyTable
import randomcolor


class ModelEvaluation:
    def __init__(self, models: ModelHistorySet):
        self._models = models
        self._filter_params = set()
        self._p_threshold = 0.01

    def set_filter_params(self, params):
        self._filter_params = set(params)
        return self

    def set_p_threshold(self, p):
        self._p_threshold = p
        return self

    def plot_distributions(self, metric, data_type: DataType, metric_func=lambda x: max(x)):
        """Plot the distribution of all models
        Get all models with the same parameter that represents multiple train runs
        Apply the metric function to get the corresponding metric from each (e.x. max accuracy)"""
        data = []
        for params, histories in self._models.same_histories(self._filter_params).items():
            for h in histories:
                data.append([params, metric_func(h.history(metric, data_type))])
        name = "{}_{}".format(data_type.name, metric)
        sns.displot(pd.DataFrame(data, columns=['type', name]),
                    x=name, hue="type", kde=True)

    def paired_statistical_test(self, metric, data_type: DataType, metric_func=lambda x: max(x)):
        models = {}
        for params, histories in self._models.same_histories(self._filter_params).items():
            for h in histories:
                data = models.get(params, [])
                data.append(metric_func(h.history(metric, data_type)))
                models[params] = data

        table = PrettyTable(["Model", "Shapiro-Wilk p-value", "Normal distributed"],
                            title="Normality of {} on {} data".format(metric, data_type.name))
        # first do a shapiro-wilk normality test for each data
        normal_model = {}
        for model, data in models.items():
            _, p = stats.shapiro(data)
            normal_model[model] = False if p < self._p_threshold else True
            table.add_row([model, "{:.4}".format(p), 'NO' if p < self._p_threshold else 'YES'])
        print(table)

        table = PrettyTable(["Model 1", "Model 2", "t-test p-value", "t-test", "Normality",
                             "Wilcoxon p-value", "Wilcoxon"],
                            title="Paired statistical tests evaluation of {} on {} data".format(metric, data_type.name))
        # now do a paired student t-test and wilcoxon signed-rank test for all pairs
        for model1, data1 in models.items():
            for model2, data2 in models.items():
                if model1 == model2:
                    continue
                _, p_s = stats.ttest_rel(data1, data2)
                _, p_w = stats.wilcoxon(data1, data2)

                table.add_row([model1, model2, "{:.4}".format(p_s),
                               'YES' if p_s < self._p_threshold else 'NO',
                               'YES' if normal_model[model1] and normal_model[model2] else 'NO',
                               "{:.4}".format(p_w), 'YES' if p_w < self._p_threshold else 'NO'])

        print(table)

    def oneway_anova_test(self, metric, data_type: DataType, metric_func=lambda x: max(x)):
        models = {}
        for params, histories in self._models.same_histories(self._filter_params).items():
            for h in histories:
                data = models.get(params, [])
                data.append(metric_func(h.history(metric, data_type)))
                models[params] = data

        # Anova requires equal variance - do Bartlet test
        _, p = stats.bartlett(*models.values())
        _, p_a = stats.f_oneway(*models.values())
        table = PrettyTable(['Bartlett p-value', 'Equal variance', 'Oneway Anova p-value', 'Equal means'],
                            title='{} on {}'.format(metric, data_type.name))
        table.add_row([p, 'NO' if p < self._p_threshold else 'YES',
                       p_a, 'NO' if p_a < self._p_threshold else 'YES'])
        print(table)

    def tukey_hsd_test(self, metric, data_type: DataType, metric_func=lambda x: max(x)):
        models = {}
        for params, histories in self._models.same_histories(self._filter_params).items():
            for h in histories:
                data = models.get(params, [])
                data.append(metric_func(h.history(metric, data_type)))
                models[params] = data

        data_arr = np.hstack(models.values())
        data_group = np.repeat(list(models.keys()), len(data_arr) / len(models))
        res = pairwise_tukeyhsd(data_arr, data_group)
        print(res)

    def plot_history(self, title, metrics: list, **params):
        data = []
        histories = self._models.filter_histories(**params)
        for metric in metrics:
            train = [h.history(metric, DataType.TRAIN) for h in histories]
            val = [h.history(metric, DataType.VALIDATION) for h in histories]
            data.append([[u, v] for u, v in zip(train, val)])
        epochs = len(data[0][0][0])

        pb = PlotBuilder().create_subplots(len(metrics), 2, fig_size=(18, 12))
        pb.set_options(color=randomcolor.RandomColor().generate(count=len(data[0])) * len(metrics))
        epoch = range(epochs)
        for i, metric in enumerate(metrics):
            pb.create_plot('Training and Validation {} - '.format(metric) + title, (epoch, 'epoch'),
                           *data[i])
        pb.show()
