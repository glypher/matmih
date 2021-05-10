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
import datetime
import time


def benchmark(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {str(datetime.timedelta(seconds=end - start))}")
        return res
    return wrapper


class ModelEvaluation:
    def __init__(self, models: ModelHistorySet):
        self._models = models
        self._filter_params = []
        self._filter_values = {}
        self._p_threshold = 0.01

    def set_filter_params(self, params:list):
        self._filter_params = params
        return self

    def set_filter_values(self, **params):
        self._filter_values = params
        return self

    def set_p_threshold(self, p):
        self._p_threshold = p
        return self

    def plot_distributions(self, metric, data_type: DataType, title, metric_func=lambda x: max(x)):
        """Plot the distribution of all models
        Get all models with the same parameter that represents multiple train runs
        Apply the metric function to get the corresponding metric from each (e.x. max accuracy)"""
        data = []
        for params, histories in self._models.same_histories(self._filter_params, self._filter_values).items():
            for h in histories:
                data.append([params, metric_func(h.history(metric, data_type))])
        name = "{}_{}".format(data_type.name, metric)
        sns.displot(pd.DataFrame(data, columns=[title, name]),
                    x=name, hue=title, kde=True)

    def results(self, metric, data_type: DataType, metric_func=lambda x: max(x)):
        """Get all models with the same parameter that represents multiple train runs
        Apply the metric function to get the corresponding metric from each (e.x. max accuracy)"""
        result = {}
        for params, histories in self._models.same_histories(self._filter_params, self._filter_values).items():
            data = []
            for h in histories:
                data.append(metric_func(h.history(metric, data_type)))
            result[params] = data

        return result

    def paired_statistical_test(self, metric, data_type: DataType, title, metric_func=lambda x: max(x), wilcoxon=False):
        models = {}
        for params, histories in self._models.same_histories(self._filter_params, self._filter_values).items():
            for h in histories:
                data = models.get(params, [])
                data.append(metric_func(h.history(metric, data_type)))
                models[params] = data

        table = PrettyTable(["Model", "Shapiro-Wilk p-value", "Normal distributed"],
                            title="Normality of {metric} on {data_type.name} data {title}")
        # first do a shapiro-wilk normality test for each data
        normal_model = {}
        for model, data in models.items():
            _, p = stats.shapiro(data)
            normal_model[model] = False if p < self._p_threshold else True
            table.add_row([model, "{:.4}".format(p), 'NO' if p < self._p_threshold else 'YES'])
        print(table)

        columns = ["Model 1", "Model 2", "p-value", "t-test", "Normal", f"Model1 vs Model2"]
        if wilcoxon:
            columns += ["Wilcoxon p-value", "Wilcoxon"]
        table = PrettyTable(columns,
                            title=f"Paired t-test statistical evaluation of {metric} on {data_type.name} data {title}")
        # now do a paired student t-test and wilcoxon signed-rank test for all pairs
        for model1, data1 in models.items():
            for model2, data2 in models.items():
                if model1 >= model2:
                    continue
                _, p_s = stats.ttest_rel(data1, data2)
                if wilcoxon:
                    _, p_w = stats.wilcoxon(data1, data2)

                data = [model1, model2, "{:.4}".format(p_s),
                               'YES' if p_s < self._p_threshold else 'NO',
                               'YES' if normal_model[model1] and normal_model[model2] else 'NO',
                              f"better" if np.average(data1) > np.average(data2) else "worse"]
                if wilcoxon:
                    data += ["{:.4}".format(p_w), 'YES' if p_w < self._p_threshold else 'NO']

                table.add_row(data)

        print(table)

    def oneway_anova_test(self, metric, data_type: DataType, title, metric_func=lambda x: max(x)):
        models = {}
        for params, histories in self._models.same_histories(self._filter_params, self._filter_values).items():
            for h in histories:
                data = models.get(params, [])
                data.append(metric_func(h.history(metric, data_type)))
                models[params] = data

        # Anova requires equal variance - do Bartlet test
        _, p = stats.bartlett(*models.values())
        _, p_a = stats.f_oneway(*models.values())
        table = PrettyTable(['Bartlett p-value', 'H0=Equal variance', 'Oneway Anova p-value', 'H0=Equal means'],
                            title=f'{metric} on {data_type.name} {title}')
        table.add_row([p, 'NO' if p < self._p_threshold else 'YES(not sure)',
                       p_a, 'NO' if p_a < self._p_threshold else 'YES(not sure)'])
        print(table)

    def tukey_hsd_test(self, metric, data_type: DataType, title, metric_func=lambda x: max(x)):
        models = {}
        for params, histories in self._models.same_histories(self._filter_params, self._filter_values).items():
            for h in histories:
                data = models.get(params, [])
                data.append(metric_func(h.history(metric, data_type)))
                models[params] = data

        data_arr = np.hstack(list(models.values()))
        data_group = np.repeat(list(models.keys()), len(data_arr) / len(models))
        res = pairwise_tukeyhsd(data_arr, data_group)
        print(res)

    def plot_history(self, title, metrics: list, **params):
        data = []
        histories = self._models.filter_histories(**params)
        for metric in metrics:
            train = [h.history(metric, DataType.TRAIN) for h in histories]
            val = [h.history(metric, DataType.VALIDATION) for h in histories]
            metric_list = []
            for u, v in zip(train, val):
                if u is not None and v is not None:
                    metric_list.append([u, v])
                elif u is None:
                    metric_list.append([v])
                elif v is None:
                    metric_list.append([u])

            data.append(metric_list)
        epochs = max([len(data[0][i][0]) for i in range(len(histories))])

        pb = PlotBuilder().create_subplots(len(metrics), 2, fig_size=(18, 12))
        pb.set_options(color=randomcolor.RandomColor().generate(count=len(data[0])) * len(metrics))
        for i, metric in enumerate(metrics):
            pb.create_plot('Training and Validation {} - '.format(metric) + title,
                           (range(epochs), 'epoch'),
                           *data[i])
        pb.show()
