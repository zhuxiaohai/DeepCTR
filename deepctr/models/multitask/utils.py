import numpy as np
import pandas as pd


def calc_lift(df, pred, target, title_name, ax, groupnum=None):
    if groupnum is None:
        groupnum = len(df.index)

    def n0(x): return sum(x == 0)
    def n1(x): return sum(x == 1)
    def total(x): return x.shape[0]
    def name(x): return '[{:.2f}'.format(x.iloc[0]) + ', ' + '{:.2f}]'.format(x.iloc[-1])
    dfkslift = df.sort_values(pred, ascending=True).reset_index(drop=True)\
        .assign(group=lambda x: np.ceil((x.index+1)/(len(x.index)/groupnum)))\
        .groupby('group').agg({target: [n0, n1, total], pred: name})\
        .reset_index().rename(columns={'name': 'range', 'n0': 'good', 'n1': 'bad', 'total': 'count'})
    columns = dfkslift.columns.droplevel(0).tolist()
    columns[0] = 'group'
    dfkslift.columns = columns
    dfkslift = dfkslift.assign(
        group=lambda x: (x.index+1)/len(x.index),
        total_distri=lambda x: x['count']/sum(x['count']),
        good_distri=lambda x: x.good/sum(x.good),
        bad_distri=lambda x: x.bad/sum(x.bad),
        cumgood_distri=lambda x: np.cumsum(x.good)/sum(x.good),
        cumbad_distri=lambda x: np.cumsum(x.bad)/sum(x.bad),
        badrate=lambda x: x.bad/(x.good+x.bad),
        cumbadrate=lambda x: np.cumsum(x.bad)/np.cumsum(x.good+x.bad),
        lift=lambda x: (np.cumsum(x.bad)/np.cumsum(x.good+x.bad))/(sum(x.bad)/sum(x.good+x.bad)))\
        .assign(ks=lambda x: abs(x.cumbad_distri-x.cumgood_distri))
    rects = ax.bar(np.arange(groupnum), dfkslift.bad_distri, width=0.3)
    ax.plot([0, groupnum-1], [1/groupnum, 1/groupnum], 'r--')
    ax.set_title(title_name)
    ax.grid(True)
    ax.set_xticks(np.arange(groupnum))
    ax.set_xticklabels(np.arange(groupnum))
    for rect, rect_value in zip(rects, dfkslift.bad_distri/dfkslift.total_distri):
        ax.annotate('{:.2f}'.format(rect_value),
                    xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    return ax


def cal_psi_score(actual_array, expected_array,
                  bins=10, quantile=True, detail=False):
    """
    :param actual_array: np.array
    :param expected_array: np.array
    :param bins: int, number_of_bins you want for calculating psi
    :param quantile: bool
    :param detail: bool, if True, print the process of calculation
    """
    # 异常处理，所有取值都相同时, 说明该变量是常量, 返回None
    if np.min(expected_array) == np.max(expected_array):
        return None
    expected_array = pd.Series(expected_array).dropna()
    actual_array = pd.Series(actual_array).dropna()

    """step1: 确定分箱间隔"""
    def scale_range(input_array, scaled_min, scaled_max):
        """
        功能: 对input_array线性放缩至[scaled_min, scaled_max]
        :param input_array: numpy array of original values, 需放缩的原始数列
        :param scaled_min: float, 放缩后的最小值
        :param scaled_max: float, 放缩后的最大值
        :return input_array: numpy array of original values, 放缩后的数列
        """
        input_array += -np.min(input_array) # 此时最小值放缩到0
        if scaled_max == scaled_min:
            raise Exception('放缩后的数列scaled_min = scaled_min, 值为{}, '
                            '请检查expected_array数值！'.format(scaled_max))
        scaled_slope = np.max(input_array) * 1.0 / (scaled_max - scaled_min)
        input_array /= scaled_slope
        input_array += scaled_min
        return input_array

    breakpoints = np.arange(0, bins + 1) / bins * 100  # 等距分箱百分比
    if not quantile:
        # 等距分箱
        breakpoints = scale_range(breakpoints,
                                  np.min(expected_array),
                                  np.max(expected_array))
    else:
        # 等频分箱
        breakpoints = np.stack([np.percentile(expected_array, b)
                                for b in breakpoints])

    """step2: 统计区间内样本占比"""
    def generate_counts(arr, breakpoints):
        """
        功能: Generates counts for each bucket by using the bucket values
        :param arr: ndarray of actual values
        :param breakpoints: list of bucket values
        :return cnt_array: counts for elements in each bucket,
                           length of breakpoints array minus one
        :return score_range_array: 分箱区间
        """
        def count_in_range(input_arr, low, high, start):
            """
            功能: 统计给定区间内的样本数(Counts elements in array between
                 low and high values)
            :param input_arr: ndarray of actual values
            :param low: float, 左边界
            :param high: float, 右边界
            :param start: bool, 取值为Ture时，区间闭合方式[low, high],否则为(low, high]
            :return cnt_in_range: int, 给定区间内的样本数
            """
            if start:
                cnt_in_range = len(np.where(np.logical_and(input_arr >= low,
                                                           input_arr <= high))[0])
            else:
                cnt_in_range = len(np.where(np.logical_and(input_arr > low,
                                                           input_arr <= high))[0])
            return cnt_in_range
        cnt_array = np.zeros(len(breakpoints) - 1)
        range_array = [''] * (len(breakpoints) - 1)
        for i in range(1, len(breakpoints)):
            cnt_array[i - 1] = count_in_range(arr,
                                              breakpoints[i - 1],
                                              breakpoints[i], i == 1)
            if 1 == i:
                range_array[i - 1] = '[' + \
                                     str(round(breakpoints[i - 1], 4)) \
                                     + ',' + str(round(breakpoints[i], 4)) \
                                     + ']'
            else:
                range_array[i - 1] = '(' + \
                                     str(round(breakpoints[i - 1], 4)) \
                                     + ',' + str(round(breakpoints[i], 4)) \
                                     + ']'

        return cnt_array, range_array

    expected_cnt, score_range_array = generate_counts(expected_array,
                                                      breakpoints)
    expected_percents = expected_cnt / len(expected_array)
    actual_cnt = generate_counts(actual_array, breakpoints)[0]
    actual_percents = actual_cnt / len(actual_array)
    delta_percents = actual_percents - expected_percents
    score_range_array = generate_counts(expected_array, breakpoints)[1]

    """step3: 得到最终稳定性指标"""
    def sub_psi(e_perc, a_perc):
        """
        功能: 计算单个分箱内的psi值
        :param e_perc: float, 期望占比
        :param a_perc: float, 实际占比
        :return value: float, 单个分箱内的psi值
        """
        if a_perc == 0: # 实际占比
            a_perc = 0.001
        if e_perc == 0: # 期望占比
            e_perc = 0.001
        value = (e_perc - a_perc) * np.log(e_perc * 1.0 / a_perc)
        return value
    sub_psi_array = [sub_psi(expected_percents[i], actual_percents[i])
                     for i in range(0, len(expected_percents))]
    if detail:
        psi_value = pd.DataFrame()
        psi_value['score_range'] = score_range_array
        psi_value['expecteds'] = expected_cnt
        psi_value['expected(%)'] = expected_percents * 100
        psi_value['actucals'] = actual_cnt
        psi_value['actucal(%)'] = actual_percents * 100
        psi_value['ac - ex(%)'] = delta_percents * 100
        psi_value['actucal(%)'] = psi_value['actucal(%)'].apply(
            lambda x: round(x, 2))
        psi_value['ac - ex(%)'] = psi_value['ac - ex(%)'].apply(
            lambda x: round(x, 2))
        psi_value['ln(ac/ex)'] = psi_value.apply(
            lambda row: np.log((row['actucal(%)'] + 0.001)
                               / (row['expected(%)'] + 0.001)), axis=1)
        psi_value['psi'] = sub_psi_array
        flag = lambda x: '<<<<<<<' if x == psi_value.psi.max() else ''
        psi_value['max'] = psi_value.psi.apply(flag)
        psi_value = psi_value.append([{'score_range': '>>> summary',
                                       'expecteds': sum(expected_cnt),
                                       'expected(%)': 100,
                                       'actucals': sum(actual_cnt),
                                       'actucal(%)': 100,
                                       'ac - ex(%)': np.nan,
                                       'ln(ac/ex)': np.nan,
                                       'psi': np.sum(sub_psi_array),
                                       'max': '<<< result'}], ignore_index=True)
    else:
        psi_value = np.sum(sub_psi_array)
    return psi_value