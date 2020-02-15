import math
import copy
import numpy as np
import pandas as pd
from functools import wraps
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier


from .stats import WOE, probability
from .merge import merge
from .utils import to_ndarray, np_count, bin_by_splits, save_json

EMPTY_BIN = -1
ELSE_GROUP = 'else'

#选择数据类型的装饰器
def support_select_dtypes(fn):

    @wraps(fn)
    def func(self, X, *args, select_dtypes = None, **kwargs):
        if select_dtypes is not None and isinstance(X, pd.DataFrame):
            X = X.select_dtypes(include = select_dtypes)

        return fn(self, X, *args, **kwargs)

    return func

#排除数据类型的装饰器
def support_exclude(fn):
    @wraps(fn)
    def func(self, X, *args, exclude = None, **kwargs):
        if exclude is not None and isinstance(X, pd.DataFrame):
            X = X.drop(columns = exclude)

        return fn(self, X, *args, **kwargs)

    return func

#保存为json格式的装饰器
def support_save_to_json(fn):
    @wraps(fn)
    def func(self, *args, to_json = None, **kwargs):
        res = fn(self, *args, **kwargs)

        if to_json is None:
            return res

        save_json(res, to_json)

    return func



class WOETransformer(TransformerMixin):
    """WOE transformer
    """
    def __init__(self):
        self.values_ = dict()
        self.woe_ = dict()
    
    #先排除不需要转换的，然后选择数据类型
    @support_exclude
    @support_select_dtypes
    def fit(self, X, y, **kwargs):
        """fit WOE transformer

        Args:
            X (DataFrame|array-like)
            y (str|array-like)
            select_dtypes (str|numpy.dtypes): `'object'`, `'number'` etc. only selected dtypes will be transform,
        """
        if not isinstance(X, pd.DataFrame):     #如果X不是df
            self.values_, self.woe_ = self._fit_woe(X, y, **kwargs)
            return self

        if isinstance(y, str):      #分离X与y,y转化为array类型
            X = X.copy()
            y = X.pop(y)

        self.values_ = dict()
        self.woe_ = dict()
        
        #逐个处理X中的特征
        for col in X:
            self.values_[col], self.woe_[col] = self._fit_woe(X[col], y)

        return self

    def _fit_woe(self, X, y):     #  X，y均为数组类型，返回的也是数组
        X = to_ndarray(X)

        values = np.unique(X)
        l = len(values)
        woe = np.zeros(l)

        for i in range(l):
            y_prob, n_prob = probability(y, mask = (X == values[i]))

            woe[i] = WOE(y_prob, n_prob)

        return values, woe


    def transform(self, X, **kwargs):
        """transform woe

        Args:
            X (DataFrame|array-like)
            default (str): 'min'(default), 'max' - the strategy to be used for unknown group

        Returns:
            array-like
        """
        if not isinstance(self.values_, dict):    #如果values_不是字典类型，此时X是数组类型
            return self._transform_apply(X, self.values_, self.woe_, **kwargs)

        res = X.copy()
        for col in X:
            if col in self.values_:     #如果col在values_的键中
                res[col] = self._transform_apply(X[col], self.values_[col], self.woe_[col], **kwargs)

        return res


    def _transform_apply(self, X, value, woe, default = 'min'):
        """transform function for single feature

        Args:
            X (array-like)
            value (array-like)
            woe (array-like)
            default (str): 'min'(default), 'max' - the strategy to be used for unknown group

        Returns:
            array-like
        """
        X = to_ndarray(X)
        res = np.zeros(len(X))

        if default is 'min':
            default = np.min(woe)
        elif default is 'max':
            default = np.max(woe)

        # replace unknown group to default value
        res[np.isin(X, value, invert = True)] = default        
        #invert=True表示对结果取反。np.isin(np.isin([1,2,3,4],[2,6],invert=True))得到[ True, False,  True,  True]

        for i in range(len(value)):
            res[X == value[i]] = woe[i]

        return res


    @support_save_to_json
    def export(self):
        if not isinstance(self.values_, dict):
            return dict(zip(self.values_, self.woe_))       #zip之后数据以tuple形式存在

        d = dict()
        for col in self.values_:
            d[col] = dict(zip(self.values_[col], self.woe_[col]))

        return d



class Combiner(TransformerMixin):
    """Combiner for merge data
    """

    def __init__(self):
        self.splits_ = dict()


    @support_exclude
    @support_select_dtypes
    def fit(self, X, y = None, **kwargs):
        """fit combiner

        Args:
            X (DataFrame|array-like): features to be combined
            y (str|array-like): target data or name of target in `X`
            method (str): the strategy to be used to merge `X`, same as `.merge`, default is `chi`
            n_bins (int): counts of bins will be combined

        Returns:
            self
        """
        if not isinstance(X, pd.DataFrame):
            self.splits_ = self._merge(X, y = y, **kwargs)    #此时X为数组形式
            return self

        if isinstance(y, str):
            X = X.copy()
            y = X.pop(y)

        self.splits_ = dict()
        for col in X:
            self.splits_[col] = self._merge(X[col], y = y, **kwargs)     #此时X为DF类型

        return self

    def _merge(self, X, y = None, method = 'chi', **kwargs):
        """merge function for fit

        Args:
            X (DataFrame|array-like): features to be combined
            y (str|array-like): target data or name of target in `X`
            method (str): the strategy to be used to merge `X`, same as `.merge`, `chi` by default

        Returns:
            array-like: array of splits
        """
        X = to_ndarray(X)

        if y is not None:
            y = to_ndarray(y)

        uni_val = False       #初始化为bool干嘛
        if not np.issubdtype(X.dtype, np.number):     #如果X不是数值类型
            # transform raw data by woe
            transer = WOETransformer()
            woe = transer.fit_transform(X, y)
            # find unique value and its woe value
            uni_val, ix_val = np.unique(X, return_index = True)
            uni_woe = woe[ix_val]
            # sort value by woe
            ix = np.argsort(uni_woe)
            # sort unique value
            uni_val = uni_val[ix]
            # replace X by sorted index
            X = self._raw_to_bin(X, uni_val)

        _, splits = merge(X, target = y, method = method, return_splits = True, **kwargs)

        return self._covert_splits(uni_val, splits)

    def transform(self, X, **kwargs):
        """transform X by combiner

        Args:
            X (DataFrame|array-like): features to be transformed
            labels (bool): if need to use labels for resulting bins, `False` by default

        Returns:
            array-like
        """
        if not isinstance(self.splits_, dict):
            return self._transform_apply(X, self.splits_, **kwargs)

        res = X.copy()
        for col in X:
            if col in self.splits_:
                res[col] = self._transform_apply(X[col], self.splits_[col], **kwargs)

        return res

    def _transform_apply(self, X, splits, labels = False):
        """transform function for single feature

        Args:
            X (array-like): feature to be transformed
            splits (array-like): splits of `X`
            labels (bool): if need to use labels for resulting bins, `False` by default

        Returns:
            array-like
        """
        X = to_ndarray(X)

        # if is not continuous      #如果splits的维度大于1或者不是数值类型
        if splits.ndim > 1 or not np.issubdtype(splits.dtype, np.number):
            bins = self._raw_to_bin(X, splits)

        else:
            if len(splits):
                bins = bin_by_splits(X, splits)
            else:
                bins = np.zeros(len(X), dtype = int)

        if labels:
            formated = self._format_splits(splits, index = True)
            empty_mask = (bins == EMPTY_BIN)
            bins = formated[bins]
            bins[empty_mask] = EMPTY_BIN

        return bins

    def _raw_to_bin(self, X, splits):
        """bin by splits

        Args:
            X (array-like): feature to be combined
            splits (array-like): splits of `X`

        Returns:
            array-like
        """
        # set default group to EMPTY_BIN
        bins = np.full(X.shape, EMPTY_BIN)       #生成指定个数同一个值组成的数组
        for i in range(len(splits)):
            group = splits[i]
            # if group is else, set all empty group to it
            if isinstance(group, str) and group == ELSE_GROUP:      #如果group=='else'，但'else'肯定是str类型啊，前面判断无必要
                bins[bins == EMPTY_BIN] = i
            else:
                bins[np.isin(X, group)] = i

        return bins

    def _format_splits(self, splits, index = False):
        l = list()
        if np.issubdtype(splits.dtype, np.number):
            sp_l = [-np.inf] + splits.tolist() + [np.inf]      #生成一个新的列表
            for i in range(len(sp_l) - 1):
                l.append('['+str(sp_l[i])+' ~ '+str(sp_l[i+1])+')')     #l类似[['2'~'5'),['5'~'8')]的样式
        else:
            for keys in splits:
                if isinstance(keys, str) and keys == ELSE_GROUP:
                    l.append(keys)
                else:
                    l.append(','.join(keys))

        if index:
            indexes = [i for i in range(len(l))]
            l = ["{}.{}".format(ix, lab) for ix, lab in zip(indexes, l)]    #生成[ix.lab]列表

        return np.array(l)

    def set_rules(self, map, reset = False):
        """set rules for combiner

        Args:
            map (dict|array-like): map of splits
            reset (bool): if need to reset combiner

        Returns:
            self
        """
        if not isinstance(map, dict):
            self.splits_ = np.array(map)
            return self

        if reset:
            self.splits_ = dict()
        
        for col in map:
            self.splits_[col] = np.array(map[col])

        return self


    @property
    def dtypes(self):
        """get the dtypes which is combiner used

        Returns:
            (str|dict)
        """
        if not isinstance(self.splits_, dict):
            return self._get_dtype(self.splits_)

        t = {}
        for n, v in self.splits_.items():
            t[n] = self._get_dtype(v)
        return t

    def _get_dtype(self, split):
        if np.issubdtype(split.dtype, np.number):
            return 'numeric'

        return 'object'


    @support_save_to_json
    def export(self, format = False):
        """export combine rules for score card

        Args:
            format (bool): if True, bins will be replace with string label for values
            to_json (str|IOBase): io to write json file

        Returns:
            dict
        """
        splits = copy.deepcopy(self.splits_)       #这里是deepcppy

        if format:
            if not isinstance(splits, dict):
                splits = self._format_splits(splits)
            else:
                for col in splits:
                    splits[col] = self._format_splits(splits[col])

        if not isinstance(splits, dict):
            bins = splits.tolist()
        else:
            bins = {k: v.tolist() for k, v in splits.items()}

        return bins


    def _covert_splits(self, value, splits):
        """covert combine rules to array
        """
        if value is False:
            return splits

        if isinstance(value, np.ndarray):
            value = value.tolist()

        start = 0
        l = list()
        for i in splits:
            i = math.ceil(i)        #向下取整
            l.append(value[start:i])    #l为二维数组 [[],[],]
            start = i

        l.append(value[start:])     

        return np.array(l)





class GBDTTransformer(TransformerMixin):
    """GBDT transformer
    """
    def __init__(self):
        self.gbdt = None
        self.onehot = None
    

    @support_exclude
    @support_select_dtypes
    def fit(self, X, y, **kwargs):
        """fit GBDT transformer

        Args:
            X (DataFrame|array-like)
            y (str|array-like)
            select_dtypes (str|numpy.dtypes): `'object'`, `'number'` etc. only selected dtypes will be transform,
        """

        if isinstance(y, str):
            X = X.copy()
            y = X.pop(y)

        self.gbdt = GradientBoostingClassifier(**kwargs)
        self.gbdt.fit(X, y)

        X = self.gbdt.apply(X)
        X = X.reshape(-1, X.shape[1])

        self.onehot = OneHotEncoder().fit(X)

        return self


    def transform(self, X):
        """transform woe

        Args:
            X (DataFrame|array-like)
            default (str): 'min'(default), 'max' - the strategy to be used for unknown group

        Returns:
            array-like
        """
        X = self.gbdt.apply(X)
        X = X.reshape(-1, X.shape[1])
        res = self.onehot.transform(X).toarray()
        return res 
