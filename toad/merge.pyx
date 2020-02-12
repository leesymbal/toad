# cython: language_level = 3, infer_types = True, boundscheck = False

import numpy as np
cimport numpy as np
cimport cython

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.cluster import KMeans
from .utils import fillna, bin_by_splits, to_ndarray, support_dataframe, clip

from cython.parallel import prange
from .c_utils cimport c_min, c_sum, c_sum_axis_0, c_sum_axis_1



DEFAULT_BINS = 10


def StepMerge(feature, nan = None, n_bins = None, clip_v = None, clip_std = None, clip_q = None):
    """Merge by step

    Args:
        feature (array-like)
        nan (number): value will be used to fill nan
        n_bins (int): n groups that will be merged into
        clip_v (number | tuple): min/max value of clipping
        clip_std (number | tuple): min/max std of clipping
        clip_q (number | tuple): min/max quantile of clipping
    Returns:
        array: split points of feature
    """
    if n_bins is None:
        n_bins = DEFAULT_BINS

    if nan is not None:
        feature = fillna(feature, by = nan)
        
    #使小于min的等于min,大于max的等于max
    feature = clip(feature, value = clip_v, std = clip_std, quantile = clip_q)
    
    #忽略nan求最大最小值
    max = np.nanmax(feature)
    min = np.nanmin(feature)

    step = (max - min) / n_bins
    return np.arange(min, max, step)[1:]      #去掉最小值

def QuantileMerge(feature, nan = -1, n_bins = None, q = None):
    """Merge by quantile

    Args:
        feature (array-like)
        nan (number): value will be used to fill nan
        n_bins (int): n groups that will be merged into
        q (array-like): list of percentage split points

    Returns:
        array: split points of feature
    """
    if n_bins is None and q is None:
        n_bins = DEFAULT_BINS
        
    #q为分位点
    if q is None:
        step = 1 / n_bins
        q = np.arange(0, 1, step)

    feature = fillna(feature, by = nan)

    splits = np.quantile(feature, q)
    return np.unique(splits)[1:]


def KMeansMerge(feature, target = None, nan = -1, n_bins = None, random_state = 1):
    """Merge by KMeans

    Args:
        feature (array-like)
        target (array-like): target will be used to fit kmeans model
        nan (number): value will be used to fill nan
        n_bins (int): n groups that will be merged into
        random_state (int): random state will be used for kmeans model

    Returns:
        array: split points of feature
    """
    if n_bins is None:
        n_bins = DEFAULT_BINS

    feature = fillna(feature, by = nan)

    model = KMeans(
        n_clusters = n_bins,
        random_state = random_state
    )
    model.fit(feature.reshape((-1 ,1)), target)     #-1表示不指定数目

    centers = np.sort(model.cluster_centers_.reshape(-1))

    l = len(centers) - 1
    splits = np.zeros(l)
    for i in range(l):
        splits[i] = (centers[i] + centers[i+1]) / 2        #前面len减去1是因为这里有i+1

    return splits



def DTMerge(feature, target, nan = -1, n_bins = None, min_samples = 1):
    """Merge continue

    Args:
        feature (array-like)
        target (array-like): target will be used to fit decision tree
        nan (number): value will be used to fill nan
        n_bins (int): n groups that will be merged into
        min_samples (int): min number of samples in each leaf nodes

    Returns:
        array: array of split points
    """
    if n_bins is None and min_samples == 1:
        n_bins = DEFAULT_BINS

    feature = fillna(feature, by = nan)

    tree = DecisionTreeClassifier(
        min_samples_leaf = min_samples,
        max_leaf_nodes = n_bins,            #最大叶子节点
    )
    tree.fit(feature.reshape((-1, 1)), target)

    thresholds = tree.tree_.threshold      #获得决策树规则
    thresholds = thresholds[thresholds != _tree.TREE_UNDEFINED]     #  _tree.TREE_UNDEFINED的值为-2
    return np.sort(thresholds)



#此两个修饰符用来关闭cython的边界检查
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef ChiMerge(feature, target, n_bins = None, min_samples = None,
            min_threshold = None, nan = -1, balance = True):
    """Chi-Merge

    Args:
        feature (array-like): feature to be merged
        target (array-like): a array of target classes
        n_bins (int): n bins will be merged into
        min_samples (number): min sample in each group, if float, it will be the percentage of samples
        min_threshold (number): min threshold of chi-square

    Returns:
        array: array of split points
    """

    # set default break condition
    if n_bins is None and min_samples is None and min_threshold is None:
        n_bins = DEFAULT_BINS

    if min_samples and min_samples < 1:
        min_samples = len(feature) * min_samples

    feature = fillna(feature, by = nan)
    target = to_ndarray(target)

    #np.unique会返回从小到大排序后的数组
    target_unique = np.unique(target)
    feature_unique = np.unique(feature)
    len_f = len(feature_unique)
    len_t = len(target_unique)

    cdef double [:,:] grouped = np.zeros((len_f, len_t), dtype=np.float)

    for r in range(len_f):
        tmp = target[feature == feature_unique[r]]    #target[]  中括号内是feature各个唯一值的索引
        for c in range(len_t):
            grouped[r, c] = (tmp == target_unique[c]).sum()      
            #grouped数组中的值为各个唯一的feature所在的索引对应的target数组与各唯一的target相等的个数


    cdef double [:,:] couple
    cdef double [:] cols, rows, chi_list
    # cdef long [:] min_ix, drop_ix
    # cdef long[:] chi_ix
    cdef double chi, chi_min, total, e
    cdef int l, retain_ix, ix
    cdef Py_ssize_t i, j, k, p

    while(True):
        # break loop when reach n_bins
        if n_bins and len(grouped) <= n_bins:     #len(grouped)即len_f.唯一特征数<=n_bins就跳出while循环
            break

        # break loop if min samples of groups is greater than threshold   
        if min_samples and c_min(c_sum_axis_1(grouped)) > min_samples:      #如果feature中特征的最小重复数大于最小样本数，跳出循环
            break

        # Calc chi square for each group
        l = len(grouped) - 1
        chi_list = np.zeros(l, dtype=np.float)
        chi_min = np.inf
        # chi_ix = []
        for i in range(l):
            chi = 0
            couple = grouped[i:i+2,:]       #取相邻两行,假如couple为[[1,2,3],[4,5,6]]
            total = c_sum(couple)
            cols = c_sum_axis_0(couple)     #cols=[5,7,9]
            rows = c_sum_axis_1(couple)     #rows=[6,15]

            for j in range(couple.shape[0]):     #等同 j   in  range(len(rows))
                for k in range(couple.shape[1]):     #等同 k  in   range(len(cols))
                    e = rows[j] * cols[k] / total
                    if e != 0:
                        chi += (couple[j, k] - e) ** 2 / e

            # balance weight of chi
            if balance:
                chi *= total

            chi_list[i] = chi

            if chi == chi_min:        #如果计算出的该相邻卡方值==chi_min，则计算下一个相邻卡方值
                chi_ix.append(i)
                continue

            if chi < chi_min:         #如果计算出的该相邻卡方值<chi_min，则令chi_min等于该卡方值
                chi_min = chi
                chi_ix = [i]

            # if chi < chi_min:
            #     chi_min = chi




        # break loop when the minimun chi greater the threshold
        if min_threshold and chi_min > min_threshold:          #跳出while循环
            break

        # get indexes of the groups who has the minimun chi
        min_ix = np.array(chi_ix)
        # min_ix = np.where(chi_list == chi_min)[0]

        # get the indexes witch needs to drop
        drop_ix = min_ix + 1                 #np.array和整数相加依然是np.array


        # combine groups by indexes
        retain_ix = min_ix[0]
        last_ix = retain_ix
        for ix in min_ix:
            # set a new group
            if ix - last_ix > 1:      #为了处理min_ix里面的相邻值
                retain_ix = ix

            # combine all contiguous indexes into one group
            for p in range(grouped.shape[1]):
                grouped[retain_ix, p] = grouped[retain_ix, p] + grouped[ix + 1, p]   
                #此处为何是ix+1,答案是为了处理min_ix里面的相邻值，配合前面的if  ix-last_ix>1

            last_ix = ix


        # drop binned groups
        grouped = np.delete(grouped, drop_ix, axis = 0)       #删除被合并的索引
        feature_unique = np.delete(feature_unique, drop_ix)


    return feature_unique[1:]

#装饰器，调用merge时实际调用的是support_dataframe.<locals>.decorator.<locals>.func(frame, *args, **kwargs). func中的fn即为下面的merge函数
@support_dataframe(require_target = False)    
def merge(feature, target = None, method = 'dt', return_splits = False, **kwargs):
    """merge feature into groups

    Args:
        feature (array-like)
        target (array-like)
        method (str): 'dt', 'chi', 'quantile', 'step', 'kmeans' - the strategy to be used to merge feature
        return_splits (bool): if needs to return splits
        n_bins (int): n groups that will be merged into


    Returns:
        array: a array of merged label with the same size of feature
        array: list of split points
    """
    feature = to_ndarray(feature)
    method = method.lower()       #将传入的method方法转为小写

    if method == 'dt':
        splits = DTMerge(feature, target, **kwargs)
    elif method == 'chi':
        splits = ChiMerge(feature, target, **kwargs)
    elif method == 'quantile':
        splits = QuantileMerge(feature, **kwargs)
    elif method == 'step':
        splits = StepMerge(feature, **kwargs)
    elif method == 'kmeans':
        splits = KMeansMerge(feature, target = target, **kwargs)
    else:
        splits = np.empty(shape = (0,))      #返回array([], dtype=float64)


    if len(splits):
        bins = bin_by_splits(feature, splits)        #bin_by_splits返回特征所在箱的位置
    else:
        bins = np.zeros(len(feature))

    if return_splits:
        return bins, splits      

    return bins
