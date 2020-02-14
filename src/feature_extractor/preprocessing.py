import numpy as np
import pandas as pd
import scipy.signal


def divide_by_max(time_series):
    y_max = np.max(time_series.y)
    if y_max > 0.001:
        time_series.y = time_series.y / y_max
    return time_series


def spike_removal(time_series, range_points=2):
    # time_series['y'] = scipy.signal.medfilt(time_series['y'], kernel_size=5)
    signal = time_series.y
    n = len(signal)
    for i in range(range_points, n - range_points - 1):
        signal[i] = np.median(signal[i - range_points : i + range_points + 1])
    time_series.y = signal
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

    return dataframes_dict, sorted(list(intersecting_times))


def get_time_delta(dataframes_dict):
    first_df = next(iter(dataframes_dict.values()))
    if len(first_df.time) < 3:
        raise ValueError('Length of dataframe time is less than 3: {}'.format(dataframes_dict))

    return int((first_df.time[2] - first_df.time[1]) / 60)


def check_length(dataframes_dict):
    first_len = len(next(iter(dataframes_dict.values())))
    if any(len(sensor_df) != first_len for sensor_df in dataframes_dict.values()):
        raise ValueError(
            'Incorrect lengths: {}'.format([len(df) for df in dataframes_dict.values()])
        )
    if any(len(sensor_df) == 0 for sensor_df in dataframes_dict.values()):
        raise ValueError('Null lengths: {}'.format([len(df) for df in dataframes_dict.values()]))


def preprocess_raw_sensor_data(raw_sensor_dataframes_dict, left_bound=None, right_bound=None):
    """Convert raw sensor dataframes from db to a dict of time_series dataframes convenient
        for analysis.

    Input:
        Dict of same length pandas dataframes with 'time' and 'avg' fields and
        name attribute.
    Preprocessing:
        Finding time delta
        Finding timestamp intersection
        Checking if all lengths are same.
        Reducing size if needed.
        Renaming.
        Dividing 'y' values by max.
        Median filtering.
        Changing index to be from 1 to len(dataframe)
    Output:
        Dict of preprocessed pandas dfs with 't' and 'y' fields and
        name attribute.
        time_delta - time interval between 't' values - used for change point detection

    """

    time_series_dict = {}

    time_delta = get_time_delta(raw_sensor_dataframes_dict)
    raw_sensor_dataframes_dict, timestamps = find_timestamp_intersection(raw_sensor_dataframes_dict)
    check_length(raw_sensor_dataframes_dict)

    sensors = list(raw_sensor_dataframes_dict.keys())
    job_data = [list(raw_sensor_dataframes_dict[sensor].avg) for sensor in sensors]
    job_data, timestamps, times, indices = reduceSizeOfData(job_data, timestamps)
    for sensor, sensor_data in zip(sensors, job_data):
        new_sensor_df_data = {'time': timestamps, 'avg': sensor_data}
        raw_sensor_dataframes_dict[sensor] = pd.DataFrame(data=new_sensor_df_data)
    time_delta *= times

    for sensor_name, raw_sensor_df in raw_sensor_dataframes_dict.items():
        # Rename
        sensor_time_series = raw_sensor_df.loc[left_bound:right_bound, ['time', 'avg']]
        sensor_time_series = sensor_time_series.rename(columns={'time': 't', 'avg': 'y'})

        # Normalize and remove spikes
        sensor_time_series = divide_by_max(sensor_time_series)
        sensor_time_series = spike_removal(sensor_time_series, 2)

        sensor_time_series.name = sensor_name
        sensor_time_series.index = np.arange(1, len(sensor_time_series) + 1)

        time_series_dict[sensor_name] = sensor_time_series
    print('Time delta {}'.format(time_delta))
    return time_series_dict, time_delta


def reduceSizeOfData(data, timestamps):
    times = 1
    if len(timestamps) <= 600:
        return data, timestamps, times, range(len(timestamps))

    indices = range(len(timestamps))
    cur_len = len(timestamps)
    tmp_data = []
    tmp_timestamps = []
    while cur_len > 600:
        # print(cur_len)
        tmp_indices = []
        for item in data:
            tmp_data.append([])
        i = 1
        while i < len(timestamps):
            if timestamps[i] - timestamps[i - 1] > 600:
                for j in range(len(data)):
                    tmp_data[j].append(data[j][i - 1])
                tmp_timestamps.append(timestamps[i - 1])
                tmp_indices.append(indices[i - 1])
                i += 1
                last_written = i - 2
                continue

            for j in range(len(data)):
                tmp_data[j].append((data[j][i - 1] + data[j][i]) / 2)
            tmp_timestamps.append(timestamps[i])
            tmp_indices.append(indices[i])
            last_written = i
            i += 2
        if last_written != len(timestamps) - 1:
            i = len(timestamps) - 1
            for j in range(len(data)):
                tmp_data[j].append(data[j][i])
            tmp_timestamps.append(timestamps[i])
            tmp_indices.append(indices[i])
        times *= 2

        data = tmp_data
        timestamps = tmp_timestamps
        tmp_data = []
        tmp_timestamps = []
        if cur_len == len(timestamps):
            break
        cur_len = len(timestamps)
        indices = tmp_indices

    return data, timestamps, times, indices
