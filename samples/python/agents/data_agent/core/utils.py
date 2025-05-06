import pandas as pd
import numpy as np

def safe_compare(series: pd.Series, operator: str, value: any) -> pd.Series:
    """
    データ型を考慮して安全に比較を行い、結果の真偽値マスクを返す。
    """
    original_value = value
    if operator == 'isna':
        return series.isna()
    if operator == 'notna':
        return series.notna()
    if operator in ['contains', 'not contains', 'startswith', 'endswith']:
        str_series = series.astype(str)
        str_value = str(value) if value is not None else ''
        try:
            if operator == 'contains':
                return str_series.str.contains(str_value, case=False, na=False)
            if operator == 'not contains':
                return ~str_series.str.contains(str_value, case=False, na=False)
            if operator == 'startswith':
                return str_series.str.startswith(str_value, na=False)
            if operator == 'endswith':
                return str_series.str.endswith(str_value, na=False)
        except Exception as e:
            raise ValueError(f"文字列操作 '{operator}' (値: {original_value}) 実行中にエラー: {e}")
    if operator in ['in', 'not in']:
        if not isinstance(value, list):
            value = [value]
        try:
            mask = series.isin(value)
            return mask if operator == 'in' else ~mask
        except Exception as e:
            raise ValueError(f"'{operator}' 操作 (値: {original_value}) 実行中にエラー: {e}")
    try:
        num_value = pd.to_numeric(value)
        value_type = 'numeric'
        compare_val = num_value
    except (ValueError, TypeError):
        try:
            dt_value = pd.to_datetime(value)
            value_type = 'datetime'
            compare_val = dt_value
        except (ValueError, TypeError, OverflowError):
            value_type = 'string'
            compare_val = str(value)
    try:
        if value_type == 'numeric':
            numeric_series = pd.to_numeric(series, errors='coerce')
            if operator == '>': return (numeric_series > compare_val).fillna(False)
            if operator == '>=': return (numeric_series >= compare_val).fillna(False)
            if operator == '<': return (numeric_series < compare_val).fillna(False)
            if operator == '<=': return (numeric_series <= compare_val).fillna(False)
            if operator == '==': return (numeric_series == compare_val).fillna(False)
            if operator == '!=': return (numeric_series != compare_val).fillna(False)
        elif value_type == 'datetime':
            datetime_series = pd.to_datetime(series, errors='coerce')
            if operator == '>': return (datetime_series > compare_val).fillna(False)
            if operator == '>=': return (datetime_series >= compare_val).fillna(False)
            if operator == '<': return (datetime_series < compare_val).fillna(False)
            if operator == '<=': return (datetime_series <= compare_val).fillna(False)
            if operator == '==': return (datetime_series == compare_val).fillna(False)
            if operator == '!=': return (datetime_series != compare_val).fillna(False)
        elif value_type == 'string':
            str_series = series.astype(str)
            if operator == '>': return str_series > compare_val
            if operator == '>=': return str_series >= compare_val
            if operator == '<': return str_series < compare_val
            if operator == '<=': return str_series <= compare_val
            if operator == '==': return str_series == compare_val
            if operator == '!=': return str_series != compare_val
        else:
            raise TypeError("予期せぬ比較値タイプです。")
    except Exception as e:
        raise ValueError(f"演算子 '{operator}' (値: {original_value}) 適用中にエラー: {e}")
    raise NotImplementedError(f"未対応または不正な演算子: {operator}") 