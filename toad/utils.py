import re
import json
import string
import numpy as np
import pandas as pd
from functools import wraps

from multiprocessing import Pool, current_process, cpu_count

#全大写变量表示常量
CONTINUOUS_NUM = 10
FEATURE_THRESHOLD = 1e-7
NAN_REPLACEMENT = -2e10

#所有可能的空值做成一个列表
NAN_LIST = [
    'nan',
    'Nan',
    'null',
    'None',
    None,
    np.nan,
]


class Parallel:
    def __init__(self):
        self.ismain = False
        self.results = []       #[]和list()是等价的
        self.pro = current_process()

        if self.pro.name == 'MainProcess':
            self.ismain = True
            self.pool = Pool(cpu_count())


    def apply(self, func, args = (), kwargs = {}):
        if not self.ismain:
            r = func(*args, **kwargs)
        else:    #注意此处的逻辑，当ismain是true时，才会有self.pool
            r = self.pool.apply_async(func, args = args, kwds = kwargs)     #r是一个实例

        self.results.append(r)

    def join(self):
        if not self.ismain:
            return self.results

        self.pool.close()
        self.pool.join()
        #实例调用get()方法返回self._value
        return [r.get() for r in self.results]


#返回数组中等于指定值的个数。如果个数为0且默认值存在，返回默认值
def np_count(arr, value, default = None):
    c = (arr == value).sum()

    if default is not None and c == 0:
        return default

    return c

#把各种空值统一替换为np.nan.
def _replace_nan(arr):
    a = np.copy(arr)      #np.copy使a的改变不影响arr
    a[a == NAN_REPLACEMENT] = np.nan
    return a


def has_nan(arr):
    return np.any(pd.isna(arr))    #np.any()  存在true就返回true    pd.isna()    返回一个布尔值数组 

#唯一化，并把nan排在前面，后面从大到小排序
def np_unique(arr, **kwargs):
    arr = to_ndarray(arr)      #首先进行转换   

    if not has_nan(arr):       #没有空值直接返回，注意np.unique默认从大到小排序
        return np.unique(arr, **kwargs)

    arr[np.isnan(arr)] = NAN_REPLACEMENT      #空值用指定值替换

    res = np.unique(arr, **kwargs)          

    if isinstance(res, tuple):
        u = _replace_nan(res[0])
        return (u, *res[1:])    
    #这两处用_replace_nan再把NAN_REPLACEMENT替换回来
    return _replace_nan(res)

#返回指定数据类型的数组
def to_ndarray(s, dtype = None):
    """
    """
    #首先针对s的类型进行处理
    if isinstance(s, np.ndarray):
        arr = np.copy(s)
    elif isinstance(s, pd.core.base.PandasObject):
        arr = np.copy(s.values)      #df.values会返回一个array
    else:
        arr = np.array(s)


    if dtype is not None:
        arr = arr.astype(dtype)
    # covert object type to str
    elif arr.dtype.type is np.object_:   #没有指定dtype且...
        arr = arr.astype(np.str)

    return arr

#指定值替换nan
def fillna(feature, by = -1):
    # copy array
    copied = np.copy(feature)

    mask = pd.isna(copied)

    copied[mask] = by

    return copied

def bin_by_splits(feature, splits):
    """Bin feature by split points
    """
    feature = fillna(feature)
    return np.digitize(feature, splits)         #返回特征位置，np.digitize([1,3,7],[2,5])得[0, 1, 2]

#按照特征值排序，相邻特征值相同略过，取不等相邻特征值的平均值为分裂点
def feature_splits(feature, target):
    """find posibility spilt points
    """
    feature = to_ndarray(feature)
    target = to_ndarray(target)

    matrix = np.vstack([feature, target])          
    matrix = matrix[:, matrix[0,:].argsort()]       #按照第一行，从小到大重新排序

    splits_values = []
    for i in range(1, len(matrix[0])):       
        # if feature value is almost same, then skip
        if matrix[0,i] <= matrix[0, i-1] + FEATURE_THRESHOLD:
            continue

        # if target value is not same, calculate split
        if matrix[1, i] != matrix[1, i-1]:
            v = (matrix[0, i] + matrix[0, i-1]) / 2.0
            splits_values.append(v)

    return np.unique(splits_values)

#生成一个有三个列且可迭代的df
def iter_df(dataframe, feature, target, splits):
    """iterate dataframe by split points

    Returns:
        iterator (df, splitter)
    """
    splits.sort()
    df = pd.DataFrame()
    df['source'] = dataframe[feature]
    df[target] = dataframe[target]
    df[feature] = 0

    for v in splits:
        df.loc[df['source'] < v, feature] = 1     #如果dataframe[feature]<v,则令df的feature=1
        yield df, v

#迭代返回由0，1组成的bin
def inter_feature(feature, splits):
    splits.sort()
    bin = np.zeros(len(feature))

    for v in splits:
        bin[feature < v] = 1
        yield bin


def is_continuous(series):
    series = to_ndarray(series)
    if not np.issubdtype(series.dtype, np.number):          #如果不是number类型，直接返回false
        return False

    n = len(np.unique(series))
    return n > CONTINUOUS_NUM or n / series.size > 0.5         #如果唯一值大于10或者唯一值大于一半，返回true
    # return n / series.size > 0.5

#分离目标变量
def split_target(frame, target):
    """
    """
    if isinstance(target, str): 
        f = frame.drop(columns = target)
        t = frame[target]
    else:
        f = frame.copy()
        t = target

    return f, t

#如果tuple只有一个元素，那么就返回这个元素
def unpack_tuple(x):
    if len(x) == 1:
        return x[0]
    else:
        return x

ALPHABET = string.ascii_uppercase + string.digits      #大写字母和数字
def generate_str(size = 6, chars = ALPHABET):
    return ''.join(np.random.choice(list(chars), size = size))


def support_dataframe(require_target = True):
    """decorator for supporting dataframe
    """
    def decorator(fn):
        @wraps(fn)       #
        def func(frame, *args, **kwargs):
            if not isinstance(frame, pd.DataFrame):
                return fn(frame, *args, **kwargs)

            frame = frame.copy()
            if require_target and isinstance(args[0], str):
                target = frame.pop(args[0])
                args = (target,) + args[1:]
            elif 'target' in kwargs and isinstance(kwargs['target'], str):
                kwargs['target'] = frame.pop(kwargs['target'])

            res = dict()
            for col in frame:
                r = fn(frame[col], *args, **kwargs)

                if not isinstance(r, np.ndarray):
                    r = [r]

                res[col] = r
            return pd.DataFrame(res)

        return func

    return decorator      #把函数作为结果返回的是嵌套函数


def save_json(contents, file, indent = 4):
    """save json file

    Args:
        contents (dict): contents to save
        file (str|IOBase): file to save
    """
    if isinstance(file, str):
        file = open(file, 'w')

    with file as f:
        json.dump(contents, f, ensure_ascii = False, indent = indent)


def read_json(file):
    """read json file
    """
    if isinstance(file, str):
        file = open(file)

    with file as f:
        res = json.load(f)

    return res



#返回np.clip(series,min,max),小于min则等于min,大于max则等于max.用于处理极端值
def clip(series, value = None, std = None, quantile = None):
    """clip series

    Args:
        series (array-like): series need to be clipped
        value (number | tuple): min/max value of clipping
        std (number | tuple): min/max std of clipping
        quantile (number | tuple): min/max quantile of clipping
    """
    series = to_ndarray(series)

    if value is not None:
        min, max = _get_clip_value(value)

    elif std is not None:
        min, max = _get_clip_value(std)
        s = np.std(series, ddof = 1)
        mean = np.mean(series)
        min = None if min is None else mean - s * min
        max = None if max is None else mean + s * max

    elif quantile is not None:
        if isinstance(quantile, tuple):
            min, max = quantile
        else:
            min = quantile
            max = 1 - quantile

        min = None if min is None else np.quantile(series, min)
        max = None if max is None else np.quantile(series, max)

    else:
        return series

    return np.clip(series, min, max)

#如果参数不是tuple类型，则返回相同两个参数
def _get_clip_value(params):
    if isinstance(params, tuple):
        return params
    else:
        return params, params

#返回日期差，以日计
def diff_time(base, target, format = None, time = 'day'):
    # if base is not a datetime list
    if not np.issubdtype(base.dtype, np.datetime64):
        base = pd.to_datetime(base, format = format, cache = True)

    target = pd.to_datetime(target, format = format, cache = True)

    delta = target - base

    if time == 'day':
        return delta.dt.days

    return delta

#返回一个只有天数差的df
def diff_time_frame(base, frame, format = None):
    res = pd.DataFrame()

    base = pd.to_datetime(base, format = format, cache = True)

    for col in frame:
        try:
            res[col] = diff_time(base, frame[col], format = format)
        except Exception as e:
            continue

    return res


def bin_to_number(reg = None):
    """
    Returns:
        function: func(string) -> number
    """
    if reg is None:
        reg = r'\d+'

    def func(x):
        if pd.isnull(x):
            return np.nan

        res = re.findall(reg, x)
        l = len(res)
        res = map(float, res)      #转化为浮点数
        if l == 0:
            return np.nan
        else:
            return sum(res) / l         #返回平均值

    return func

#为拒绝推断生成目标
def generate_target(size, rate = 0.5, weight = None, reverse = False):
    """generate target for reject inference

    Args:
        size (int): size of target
        rate (float): rate of '1' in target
        weight (array-like): weight of '1' to generate target
        reverse (bool): if need reverse weight

    Returns:
        array
    """
    if weight is not None:
        weight = np.asarray(weight)

        if reverse is True:
            weight = (np.max(weight) + np.min(weight)) - weight

        weight = weight / weight.sum()

    res = np.zeros(size)

    choice_num = int(size * rate)
    ix = np.random.choice(size, choice_num, replace = False, p = weight)        
    #size如果是整数表示np.arange(size),choice_num表示要选择的个数，replace表示能否重复选取一个值
    res[ix] = 1

    return res


def get_dummies(dataframe, exclude = None, binary_drop = False, **kwargs):
    """get dummies
    """
    #找出非数值型列
    columns = dataframe.select_dtypes(exclude = 'number').columns

    if len(columns) == 0:
        return dataframe

    if exclude is not None:
        columns = columns.difference(exclude)       #difference为减去

    if binary_drop:
        mask = dataframe[columns].nunique(dropna = False) == 2     #在非数值型列中找出唯一值为2的列，nunique.返回布尔数组 

        if mask.sum() != 0:
            dataframe = pd.get_dummies(
                dataframe,
                columns = columns[mask],     #选出二值型列
                drop_first = True,          #true则get_dummies后该列仍为一列
                **kwargs,
            )
            columns = columns[~mask]        #此处选出非二值型列

    data = pd.get_dummies(dataframe, columns = columns, **kwargs)
    return data
