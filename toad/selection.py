import numpy as np
import pandas as pd
from scipy import stats

from .stats import IV, VIF
from .metrics import MSE, AIC, BIC, KS, AUC
from .utils import split_target, unpack_tuple, to_ndarray


INTERCEPT_COLS = 'intercept'


class StatsModel:
    def __init__(self, estimator = 'ols', criterion = 'aic', intercept = False):
        if isinstance(estimator, str):
            Est = self.get_estimator(estimator)
            estimator = Est(fit_intercept = intercept,)        #intercept_=False表示截距为0 

        self.estimator = estimator
        self.intercept = intercept
        self.criterion = criterion


    def get_estimator(self, name):
        from sklearn.linear_model import (
            LinearRegression,
            LogisticRegression,
            Lasso,
            Ridge,
        )

        ests = {                     #将可选模型组装为字典
            'ols': LinearRegression,
            'lr': LogisticRegression,
            'lasso': Lasso,
            'ridge': Ridge,
        }

        if name in ests:
            return ests[name]
        #此处无须用else语句了 
        raise Exception('estimator {name} is not supported'.format(name = name))



    def stats(self, X, y):
        """
        """
        X = X.copy()

        if isinstance(X, pd.Series):
            X = X.to_frame()           #frame比起series多了列名

        self.estimator.fit(X, y)

        if hasattr(self.estimator, 'predict_proba'):       #如果模型支持输出概率
            pre = self.estimator.predict_proba(X)[:, 1]
        else:
            pre = self.estimator.predict(X)

        coef = self.estimator.coef_.reshape(-1)

        if self.intercept:
            coef = np.append(coef, self.estimator.intercept_)
            X[INTERCEPT_COLS] = np.ones(X.shape[0])

        n, k = X.shape

        t_value = self.t_value(pre, y, X, coef)
        p_value = self.p_value(t_value, n)
        c = self.get_criterion(pre, y, k)

        return {
            't_value': pd.Series(t_value, index = X.columns),
            'p_value': pd.Series(p_value, index = X.columns),
            'criterion': c
        }

    def get_criterion(self, pre, y, k):
        if self.criterion == 'aic':
            llf = self.loglikelihood(pre, y, k)
            return AIC(pre, y, k, llf = llf)

        if self.criterion == 'bic':
            llf = self.loglikelihood(pre, y, k)
            return BIC(pre, y, k, llf = llf)

        if self.criterion == 'ks':
            return KS(pre, y)

        if self.criterion == 'auc':
            return AUC(pre, y)

    def t_value(self, pre, y, X, coef):
        n, k = X.shape
        mse = sum((y - pre) ** 2) / float(n - k)
        nx = np.dot(X.T, X)           #X是二维数组的话，np.dot则输出矩阵。如果是一维数组，则输出数值

        if np.linalg.det(nx) == 0:    #计算行列式值
            return np.nan

        std_e = np.sqrt(mse * (np.linalg.inv(nx).diagonal()))    #np.linalg.inv矩阵求逆，diagonal()给出对角线元素
        return coef / std_e

    def p_value(self, t, n):
        return stats.t.sf(np.abs(t), n - 1) * 2

    def loglikelihood(self, pre, y, k):
        n = len(y)
        mse = MSE(pre, y)
        return (-n / 2) * np.log(2 * np.pi * mse * np.e)


def stepwise(frame, target = 'target', estimator = 'ols', direction = 'both', criterion = 'aic',
            p_enter = 0.01, p_remove = 0.01, p_value_enter = 0.2, intercept = False,
            max_iter = None, return_drop = False, exclude = None):
    """stepwise to select features

    Args:
        frame (DataFrame): dataframe that will be use to select
        target (str): target name in frame
        estimator (str): model to use for stats
        direction (str): direction of stepwise, support 'forward', 'backward' and 'both', suggest 'both'
        criterion (str): criterion to statistic model, support 'aic', 'bic'
        p_enter (float): threshold that will be used in 'forward' and 'both' to keep features     #保留特征的阀值
        p_remove (float): threshold that will be used in 'backward' to remove features
        intercept (bool): if have intercept
        p_value_enter (float): threshold that will be used in 'both' to remove features
        max_iter (int): maximum number of iterate
        return_drop (bool): if need to return features' name who has been dropped
        exclude (array-like): list of feature names that will not be dropped

    Returns:
        DataFrame: selected dataframe
        array: list of feature names that has been dropped
    """
    df, y = split_target(frame, target)

    if exclude is not None:
        df = df.drop(columns = exclude)

    drop_list = []
    remaining = df.columns.tolist()     #如果没有tolist(),只是index对象

    selected = []

    sm = StatsModel(estimator = estimator, criterion = criterion, intercept = intercept)

    order = -1 if criterion in ['aic', 'bic'] else 1

    best_score = -np.inf * order

    iter = -1
    while remaining:
        iter += 1
        if max_iter and iter > max_iter:     #迭代次数大于阀值则退出remaining
            break

        l = len(remaining)
        test_score = np.zeros(l)
        test_res = np.empty(l, dtype = np.object)      #生成的是None组成的数组

        if direction is 'backward':
            for i in range(l):
                test_res[i] = sm.stats(
                    df[ remaining[:i] + remaining[i+1:] ],     #除去第i个特征
                    y,
                )
                test_score[i] = test_res[i]['criterion']

            curr_ix = np.argmax(test_score * order)      #np.argmax返回最大值所在的位置
            curr_score = test_score[curr_ix]

            if (curr_score - best_score) * order < p_remove:
                break                                       #此处退出的是while循环

            name = remaining.pop(curr_ix)        #删除该位置的特征并赋值给name
            drop_list.append(name)

            best_score = curr_score              

        # forward and both
        else:
            for i in range(l):
                test_res[i] = sm.stats(
                    df[ selected + [remaining[i]] ],       # 已有特征加第i个特征，注意remaining[i]外面必须有中括号
                    y,
                )
                test_score[i] = test_res[i]['criterion']

            curr_ix = np.argmax(test_score * order)
            curr_score = test_score[curr_ix]

            name = remaining.pop(curr_ix)
            if (curr_score - best_score) * order < p_enter:
                drop_list.append(name)

                # early stop
                if selected:
                    drop_list += remaining
                    break             #跳出while循环

                continue              #继续while循环

            selected.append(name)
            best_score = curr_score

            if direction is 'both':
                p_values = test_res[curr_ix]['p_value']
                drop_names = p_values[p_values > p_value_enter].index

                for name in drop_names:
                    selected.remove(name)
                    drop_list.append(name)
    #（此处已在while循环外）
    r = frame.drop(columns = drop_list)

    res = (r,)
    if return_drop:
        res += (drop_list,)

    return unpack_tuple(res)


def drop_empty(frame, threshold = 0.9, nan = None, return_drop = False,
            exclude = None):
    """drop columns by empty

    Args:
        frame (DataFrame): dataframe that will be used
        threshold (number): drop the features whose empty num is greater than threshold. if threshold is float, it will be use as percentage
        nan (any): values will be look like empty
        return_drop (bool): if need to return features' name who has been dropped
        exclude (array-like): list of feature names that will not be dropped

    Returns:
        DataFrame: selected dataframe
        array: list of feature names that has been dropped
    """
    df = frame.copy()

    if exclude is not None:
        df = df.drop(columns = exclude)

    if nan is not None:
        df = df.replace(nan, np.nan)            #将nan表示的值替换为np.nan

    if threshold < 1:
        threshold = len(df) * threshold         #len(df)得到df的特征长度

    drop_list = []
    for col in df:
        n = df[col].isnull().sum()
        if n > threshold:
            drop_list.append(col)

    r = frame.drop(columns = drop_list)

    res = (r,)
    if return_drop:
        res += (np.array(drop_list),)

    return unpack_tuple(res)


def drop_var(frame, threshold = 0, return_drop = False, exclude = None):
    """drop columns by variance

    Args:
        frame (DataFrame): dataframe that will be used
        threshold (float): drop features whose variance is less than threshold
        return_drop (bool): if need to return features' name who has been dropped
        exclude (array-like): list of feature names that will not be dropped

    Returns:
        DataFrame: selected dataframe
        array: list of feature names that has been dropped
    """
    df = frame.copy()

    if exclude is not None:
        df = df.drop(columns = exclude)

    # numeric features only
    df = df.select_dtypes(include = 'number')       #数值型特征才有方差

    variances = np.var(df, axis = 0)                #分别计算各个特征的方差
    drop_list = df.columns[variances <= threshold]

    r = frame.drop(columns = drop_list)

    res = (r,)
    if return_drop:
        res += (drop_list)

    return unpack_tuple(res)


def drop_corr(frame, target = None, threshold = 0.7, by = 'IV',
            return_drop = False, exclude = None):
    """drop columns by correlation

    Args:
        frame (DataFrame): dataframe that will be used
        target (str): target name in dataframe
        threshold (float): drop features that has the smallest weight in each groups whose correlation is greater than threshold
        by (array-like): weight of features that will be used to drop the features
        return_drop (bool): if need to return features' name who has been dropped
        exclude (array-like): list of feature names that will not be dropped

    Returns:
        DataFrame: selected dataframe
        array: list of feature names that has been dropped
    """
    if not isinstance(by, (str, pd.Series)):      #如果by不是字符串或者数组类型，则转为pandas数组类型
        by = pd.Series(by, index = frame.columns)

    df = frame.copy()

    if exclude is not None:
        exclude = exclude if isinstance(exclude, (list, np.ndarray)) else [exclude]    #exclude需要转为数组形式
        df = df.drop(columns = exclude)


    f, t = split_target(df, target)

    corr = f.corr().abs()

    drops = []

    # get position who's corr greater than threshold
    ix, cn = np.where(np.triu(corr.values, 1) > threshold)     #np.triu(m,k)返回一个上三角矩阵,k越大，则0越多.这里ix是符合条件的行索引，cn为列

    # if has position
    if len(ix):
        # get the graph of relationship
        graph = np.hstack([ix.reshape((-1, 1)), cn.reshape((-1, 1))])

        uni, counts = np.unique(graph, return_counts = True)

        # calc weights for nodes
        weights = np.zeros(len(corr.index))


        if isinstance(by, pd.Series):
            weights = by[corr.index].values
        elif by.upper() == 'IV':
            for ix in uni:
                weights[ix] = IV(df[corr.index[ix]], target = t)


        while(True):
            # TODO deal with circle

            # get nodes with the most relationship
            nodes = uni[np.argwhere(counts == np.amax(counts))].flatten()

            # get node who has the min weights
            n = nodes[np.argsort(weights[nodes])[0]]

            # get nodes of 1 degree relationship of n
            i, c = np.where(graph == n)
            pairs = graph[(i, 1-c)]

            # if sum of 1 degree nodes greater than n
            # then delete n self
            # else delete all 1 degree nodes
            if weights[pairs].sum() > weights[n]:
                dro = [n]
            else:
                dro = pairs.tolist()

            # add nodes to drops list
            drops += dro

            # delete nodes from graph
            di, _ = np.where(np.isin(graph, dro))
            graph = np.delete(graph, di, axis = 0)

            # if graph is empty
            if len(graph) <= 0:
                break

            # update nodes and counts
            uni, counts = np.unique(graph, return_counts = True)


    drop_list = corr.index[drops].values
    r = frame.drop(columns = drop_list)

    res = (r,)
    if return_drop:
        res += (drop_list,)

    return unpack_tuple(res)


def drop_iv(frame, target = 'target', threshold = 0.02, return_drop = False,
            return_iv = False, exclude = None):
    """drop columns by IV

    Args:
        frame (DataFrame): dataframe that will be used
        target (str): target name in dataframe
        threshold (float): drop the features whose IV is less than threshold
        return_drop (bool): if need to return features' name who has been dropped
        return_iv (bool): if need to return features' IV
        exclude (array-like): list of feature names that will not be dropped

    Returns:
        DataFrame: selected dataframe
        array: list of feature names that has been dropped
        Series: list of features' IV
    """
    df = frame.copy()

    if exclude is not None:
        df = df.drop(columns = exclude)

    f, t = split_target(df, target)

    l = len(f.columns)        #返回特征数，注意len(f)返回特征长度
    iv = np.zeros(l)

    for i in range(l):
        iv[i] = IV(f[f.columns[i]], target = t)

    drop_ix = np.where(iv < threshold)

    drop_list = f.columns[drop_ix].values
    df = frame.drop(columns = drop_list)

    res = (df,)
    if return_drop:
        res += (drop_list,)

    if return_iv:
        res += (pd.Series(iv, index = f.columns),)

    return unpack_tuple(res)


def drop_vif(frame, threshold = 3, return_drop = False, exclude = None):
    """variance inflation factor

    Args:
        frame (DataFrame)
        threshold (float): drop features until all vif is less than threshold
        return_drop (bool): if need to return features' name who has been dropped
        exclude (array-like): list of feature names that will not be dropped

    Returns:
        DataFrame: selected dataframe
        array: list of feature names that has been dropped
    """
    df = frame.copy()

    if exclude is not None:
        df = df.drop(columns = exclude)

    drop_list = []
    while(True):
        vif = VIF(df)

        ix = vif.idxmax()
        max = vif[ix]

        if max < threshold:
            break

        df = df.drop(columns = ix)
        drop_list.append(ix)


    r = frame.drop(columns = drop_list)

    res = (r,)
    if return_drop:
        res += (drop_list,)

    return unpack_tuple(res)


def select(frame, target = 'target', empty = 0.9, iv = 0.02, corr = 0.7,
            return_drop = False, exclude = None):
    """select features by rate of empty, iv and correlation

    Args:
        frame (DataFrame)
        target (str): target's name in dataframe
        empty (number): drop the features which empty num is greater than threshold. if threshold is float, it will be use as percentage
        iv (float): drop the features whose IV is less than threshold
        corr (float): drop features that has the smallest IV in each groups which correlation is greater than threshold
        return_drop (bool): if need to return features' name who has been dropped
        exclude (array-like): list of feature name that will not be dropped

    Returns:
        DataFrame: selected dataframe
        dict: list of dropped feature names in each step
    """
    empty_drop = iv_drop = corr_drop = None

    if empty is not False:
        frame, empty_drop = drop_empty(frame, threshold = empty, return_drop = True, exclude = exclude)

    if iv is not False:
        frame, iv_drop, iv_list = drop_iv(frame, target = target, threshold = iv, return_drop = True, return_iv = True, exclude = exclude)

    if corr is not False:
        weights = 'IV'

        if iv is not False:     #如果iv存在，则weights=iv_list
            weights = iv_list

        frame, corr_drop = drop_corr(frame, target = target, threshold = corr, by = weights, return_drop = True, exclude = exclude)

    res = (frame,)
    if return_drop:
        d = {                   #所有因为empty,iv,corr删除的特征组成一个字典
            'empty': empty_drop,
            'iv': iv_drop,
            'corr': corr_drop,
        }
        res += (d,)

    return unpack_tuple(res)
