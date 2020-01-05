import numpy as np
import scipy.signal


def min_max_normalize(time_series):
    return (time_series - time_series.min()) / (time_series.max() - time_series.min())


def mean_normalize(time_series):
    return (time_series - time_series.mean()) / (time_series.max() - time_series.min())


def standartize(time_series):
    return (time_series - time_series.mean()) / time_series.std()


def median_filtration(time_series):
    time_series['y'] = scipy.signal.medfilt(time_series['y'], kernel_size=3)
    return time_series


def preprocess_raw_sensor_data(raw_sensor_dataframes_list, left_bound=None, right_bound=None):
    """Convert raw sensor dataframes from db to a list of time_series dataframes convenient
        for analysis.

    Input:
        List of same length pandas dataframes with 'time' and 'avg' fields and
        name attribute.
    Preprocessing:
        Choosing and renaming columns.
        Min-max normalization both in 't' and 'y'.
        Median filtering with kernel size 3.
        Changing index to be from 1 to len(dataframe)
    Output:
        List of preprocessed pandas dfs with 't' and 'y' fields and
        name attribute.

        Note: Input dataframes must have 'name' attribute.
    """
    assert all(len(df) == len(raw_sensor_dataframes_list[0]) for df in raw_sensor_dataframes_list)

    time_series_list = []
    for raw_sensor_df in raw_sensor_dataframes_list:
        sensor_time_series = raw_sensor_df.loc[left_bound:right_bound, ['time', 'avg']]
        sensor_time_series = sensor_time_series.rename(columns={'time': 't', 'avg': 'y'})
        sensor_time_series = min_max_normalize(sensor_time_series)
        sensor_time_series = median_filtration(sensor_time_series)
        sensor_time_series.name = raw_sensor_df.name
        sensor_time_series.index = np.arange(1, len(sensor_time_series) + 1)
        time_series_list.append(sensor_time_series)
    return time_series_list
