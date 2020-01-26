from opster import command
import os
import pandas as pd
import pickle

import config.config as config


def read_job_data_by_sensor(job_files_by_sensor):
    """For each sensor in dict converts value from list of filenames to list of pandas Dataframes"""
    job_data_by_sensor = {}
    for sensor_name, sensor_files in job_files_by_sensor.items():
        job_data_by_sensor[sensor_name] = [
            pd.read_csv(filepath, dtype=float) for filepath in sensor_files
        ]
    return job_data_by_sensor


def validate_job_data_by_sensor(job_data_by_sensor):
    """For each sensor in dict make sure that indexes match"""
    for sensor_name, sensor_dataframes in job_data_by_sensor.items():
        shapes = [df.shape for df in sensor_dataframes]
        if not all(shape == shapes[0] for shape in shapes):
            '''print(
                'Not all node data for sensor "{}" have same shape, shapes {}. '
                'Please, reload job from db.'.format(sensor_name, shapes)
            )'''
            # exit(1)


def average_job_data_by_node(job_data_by_sensor):
    averaged_job_data = {}
    for sensor_name, sensor_dataframes in job_data_by_sensor.items():
        concatenated_dataframes = pd.concat(sensor_dataframes)
        by_row_index = concatenated_dataframes.groupby(concatenated_dataframes.index)
        averaged_job_data[sensor_name] = by_row_index.mean()
        averaged_job_data[sensor_name].name = sensor_name
    return averaged_job_data


@command()
def main(job_id):
    """ This script does the following:
        1. Open job_data_db directory  (data/raw_data/job<job_id>/) with
        many csv's (one per node and sensor)
        2. Check if all sensors have the same number of nodes.
        3. Average data by node and save to
        data/node_average/job<job_id>/ directory as
        one pickle file with a dict of pandas dataframes.
        """
    job_raw_data_dir_path = '../../data/raw_data/job{}'.format(job_id)
    job_files = os.listdir(job_raw_data_dir_path)

    job_files_by_sensor = {}
    for sensor_name in config.SENSORS_LIST:
        sensor_filenames = [filename for filename in job_files if sensor_name in filename]
        if sensor_filenames:
            job_files_by_sensor[sensor_name] = [
                '{}/{}'.format(job_raw_data_dir_path, name) for name in sensor_filenames
            ]

    if len(job_files_by_sensor) < len(config.SENSORS_LIST):
        print('Not all sensors from config present. Please, reload job from db.')
        exit(1)

    sensors_count = [len(sensors_list) for sensors_list in job_files_by_sensor.values()]
    if not all(count == sensors_count[0] for count in sensors_count):
        print('Not all sensors have same number of nodes. Please, reload job from db.')
        exit(1)

    job_data_by_sensor = read_job_data_by_sensor(job_files_by_sensor)
    validate_job_data_by_sensor(job_data_by_sensor)
    averaged_job_data = average_job_data_by_node(job_data_by_sensor)

    node_average_data_file_path = '../../data/node_average_data/job{}.pickle'.format(job_id)
    with open(node_average_data_file_path, 'wb+') as pickle_file:
        pickle.dump(averaged_job_data, pickle_file)


if __name__ == '__main__':
    main.command()
