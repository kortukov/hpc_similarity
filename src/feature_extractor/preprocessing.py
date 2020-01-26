import numpy as np
import scipy.signal


def divide_by_max(time_series):
    y_max = time_series.y.max()
    if y_max > 0.001:
        time_series.y = time_series.y / y_max
    return time_series


def median_filtration(time_series):
    time_series['y'] = scipy.signal.medfilt(time_series['y'], kernel_size=5)
    return time_series


def find_timestamp_intersection(dataframes_dict):
    first_df = next(iter(dataframes_dict.values()))
    intersecting_times = set(first_df.time)
    for sensor_name in dataframes_dict:
        df = dataframes_dict[sensor_name]
        intersecting_times = intersecting_times.intersection(set(df.time))

    for sensor_name in dataframes_dict:
        df = dataframes_dict[sensor_name]
        dataframes_dict[sensor_name] = df[df.time.isin(intersecting_times)]

    return dataframes_dict


def get_time_delta(dataframes_dict):
    first_df = next(iter(dataframes_dict.values()))
    if len(first_df.time) < 2:
        raise ValueError('Length of dataframe time is less than 2')

    return int((first_df.time[1] - first_df.time[0]) / 60)


def preprocess_raw_sensor_data(raw_sensor_dataframes_dict, left_bound=None, right_bound=None):
    """Convert raw sensor dataframes from db to a dict of time_series dataframes convenient
        for analysis.

    Input:
        Dict of same length pandas dataframes with 'time' and 'avg' fields and
        name attribute.
    Preprocessing:
        Finding timestamp intersection
        Choosing and renaming columns.
        Dividing 'y' values by max.
        Median filtering with kernel size 5.
        Changing index to be from 1 to len(dataframe)
    Output:
        Dict of preprocessed pandas dfs with 't' and 'y' fields and
        name attribute.
        time_delta - time interval between 't' values - used for change point detection

    """

    time_series_dict = {}

    raw_sensor_dataframes_dict = find_timestamp_intersection(raw_sensor_dataframes_dict)
    time_delta = get_time_delta(raw_sensor_dataframes_dict)
    for sensor_name, raw_sensor_df in raw_sensor_dataframes_dict.items():
        sensor_time_series = raw_sensor_df.loc[left_bound:right_bound, ['time', 'avg']]
        sensor_time_series = sensor_time_series.rename(columns={'time': 't', 'avg': 'y'})
        sensor_time_series = divide_by_max(sensor_time_series)
        sensor_time_series = median_filtration(sensor_time_series)
        sensor_time_series.name = sensor_name
        sensor_time_series.index = np.arange(1, len(sensor_time_series) + 1)
        time_series_dict[sensor_name] = sensor_time_series
    return time_series_dict, time_delta
