import pandas as pd
import numpy as np
from sklearn.preprocessing import TargetEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression


def target_encode(data: pd.DataFrame, column_name: str, target: pd.Series=None, encoder=None, random_seed=42):
    data_encoded = data.copy()
    if encoder is None:
        encoder = TargetEncoder(target_type='continuous', random_state=random_seed)
        data_encoded.loc[:, column_name] = encoder.fit_transform(data_encoded[[column_name]], target)
        return data_encoded, encoder
    else:
        data_encoded.loc[:, column_name] = encoder.transform(data_encoded[[column_name]])
        return data_encoded


def imputer_KNN(data: pd.DataFrame, in_section_neighbors=5, cross_section_neighbors=10):
    grouped = data.groupby('date')
    in_section_imputer = KNNImputer(n_neighbors=in_section_neighbors)
    data_filled = data.copy()
    all_nan_date = 0

    for date, group in grouped:      
        # 检查是否存在全是 NaN 的列
        nan_columns = group.columns[group.isna().all()]
        ignored_columns = ['code', 'date']
        
        if not nan_columns.empty:
            ignored_columns = ignored_columns + list(nan_columns)
            all_nan_date += 1
            print('all NaN columns detected')
            print(date)
            print(nan_columns)
            #print(ignored_columns)
            
        cols_to_fill = [col for col in group.columns if col not in ignored_columns]  # 获取截面上需要填充的列
        #print(cols_to_fill)
        data_filled.loc[group.index, cols_to_fill] = in_section_imputer.fit_transform(group[cols_to_fill])
        
    # 使用整个填补完其他列的数据集进行 KNN 填补那些当天全是 NaN 的列
    if all_nan_date > 0:
        print('有', all_nan_date, '天存在部分属性全为NaN。')
        cross_section_imputer = KNNImputer(n_neighbors=cross_section_neighbors)
        ignored_columns = ['code', 'date']
        cols_to_fill = [col for col in group.columns if col not in ignored_columns]  # 获取需要标准化的列
        data_filled.loc[:, cols_to_fill] = cross_section_imputer.fit_transform(data_filled.loc[:, cols_to_fill])

    return data_filled
    
    """
    grouped = data.groupby('date')
    ingroup_imputer = KNNImputer(n_neighbors=in_section_neighbors)
    data_filled = pd.DataFrame()
    all_nan_date = 0
    
    for date, group in grouped:      
        # 检查是否存在全是 NaN 的列
        nan_columns = group.columns[group.isna().all()]
        if not nan_columns.empty:
            group = group.drop(columns=nan_columns)  # 删除全是 NaN 的列
            all_nan_date += 1
            print('all NaN columns detected')
            print(date)
            print(nan_columns)
        
        tmp_code = group['code']
        filled_group = pd.DataFrame(ingroup_imputer.fit_transform(group.drop(columns=['code'])), columns=group.columns[1:], index=group.index)
        filled_group.insert(0, 'code', tmp_code)
        data_filled = pd.concat([data_filled, filled_group], ignore_index=True)

    # 使用整个填补完其他列的数据集进行 KNN 填补那些全是 NaN 的列
    if all_nan_date > 0:
        print('有', all_nan_date, '天存在部分属性全为NaN。')
        outgroup_imputer = KNNImputer(n_neighbors=cross_section_neighbors)
        tmp_code = data_filled['code']
        data_filled = pd.DataFrame(outgroup_imputer.fit_transform(data_filled.drop(columns=['code'])), columns=data_filled.columns[1:], index=data_filled.index)
        data_filled.insert(0, 'code', tmp_code)

    return data_filled
    """


def winsorize_X(data: pd.DataFrame, lower_bound=None, upper_bound=None):
    data_winsorized = data.copy()
    lower = {}
    upper = {}
    if lower_bound is None and upper_bound is None:
        for col in data_winsorized.columns:
            if col not in ['code', 'date', 'f_6']:
                # 计算分位数时忽略 NaN 值
                q_99 = data_winsorized[col].quantile(0.99)
                q_1 = data_winsorized[col].quantile(0.01)
                #print(col)
                #print(q_99)
                #print(q_1)
        
                # 使用 clip 函数对列进行缩尾处理
                data_winsorized[col] = np.clip(data_winsorized[col], q_1, q_99)

                #存储训练集的0.01和0.99分位数
                lower[col] = q_1
                upper[col] = q_99

        return data_winsorized, lower, upper
    else:
        for col in data_winsorized.columns:
            if col not in ['code', 'date', 'f_6']:
                # 使用 clip 函数对列进行缩尾处理，使用训练集的0.01和0.99分位数
                data_winsorized[col] = np.clip(data_winsorized[col], lower_bound[col], upper_bound[col])

        return data_winsorized


def winsorize_MAD(data: pd.DataFrame, lower_bound=None, upper_bound=None):
    data_winsorized = data.copy()
    lower = {}
    upper = {}
    if lower_bound is None and upper_bound is None:
        n = 3
        for col in data_winsorized.columns:
            if col not in ['code', 'date', 'f_6']:
                # 计算分位数时忽略 NaN 值
                median = data_winsorized[col].quantile(0.5)
                mad = ((data_winsorized[col] - median).abs()).quantile(0.5)
                max_range = median + n * mad
                min_range = median - n * mad
       
                # 使用 clip 函数对列进行缩尾处理
                data_winsorized[col] = np.clip(data_winsorized[col], min_range, max_range)

                #存储训练集的上下3倍MAD
                lower[col] = min_range
                upper[col] = max_range

        return data_winsorized, lower, upper
    else:
        for col in data_winsorized.columns:
            if col not in ['code', 'date', 'f_6']:
                # 使用 clip 函数对列进行缩尾处理，使用训练集的上下3倍MAD
                data_winsorized[col] = np.clip(data_winsorized[col], lower_bound[col], upper_bound[col])

        return data_winsorized

    
def zscore_standardization(data: pd.DataFrame):
    data_standardized = data.copy()
    scaler = StandardScaler()
    date_grouped = data.groupby('date')
    cols_to_standardize = [col for col in data.columns if col not in ['code', 'date']]  # 获取截面上需要标准化的列
    
    for date, group in date_grouped:        
        data_standardized.loc[group.index, cols_to_standardize] = scaler.fit_transform(group[cols_to_standardize])
    
    return data_standardized


def robust_zscore(data: pd.DataFrame, MEDIANS=None, MADS=None):#在整个数据集上用robust zscore标准化
    data_standardized = data.copy()
    med_dict = {}
    mad_dict = {}
    if MEDIANS is None and MADS is None:
        for col in data_standardized.columns:
            if col not in ['code', 'date']:
                # 计算分位数时忽略 NaN 值
                median = data_standardized[col].quantile(0.5)
                mad = ((data_standardized[col] - median).abs()).quantile(0.5)
       
                # 按列使用R.Z.score转换
                data_standardized[col] = 0.6745 * (data_standardized[col] - median) / mad

                #存储训练集的上下3倍MAD
                med_dict[col] = median
                mad_dict[col] = mad

        return data_standardized, med_dict, mad_dict
    else:
        for col in data_standardized.columns:
            if col not in ['code', 'date']:
                # 按列使用训练集的中位数和MAD进行R.Z.score转换
                data_standardized[col] = 0.6745 * (data_standardized[col] - MEDIANS[col]) / MADS[col]

        return data_standardized


def mutual_info_selection(data: pd.DataFrame, target: pd.Series=None, selector=None, feature_num: int=10):
    if selector is None:
        selector = SelectKBest(mutual_info_regression, k=feature_num)
        X_selected = selector.fit_transform(data.iloc[:, 2:], target)
        data_selected = data.loc[:, ['code', 'date'] + list(selector.get_feature_names_out())]
        return data_selected, selector
    else:
        X_selected = selector.transform(data.iloc[:, 2:])
        data_selected = data.loc[:, ['code', 'date'] + list(selector.get_feature_names_out())]
        return data_selected