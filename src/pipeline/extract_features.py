from functools import partial
from opster import command
from multiprocessing import Pool
import pickle
import sys
import time

sys.path.append('../')
import feature_extractor.preprocessing as preprocessing
import feature_extractor.change_point_detection as cpd
import feature_extractor.feature_extraction as extraction


def extract_superstructure_wrapper(sensor_and_df_tuple, change_points):
    sensor, df = sensor_and_df_tuple
    superstructure = extraction.detect_superstructure(df, change_points=change_points)
    return sensor, superstructure


def parallel_extract_superstructure(preprocessed_job_data, change_points):
    extract_func = partial(extract_superstructure_wrapper, change_points=change_points)
    pool = Pool(2)
    job_superstructure_list = pool.map(extract_func, preprocessed_job_data.items())
    job_superstructure_dict = {
        sensor: superstructure for sensor, superstructure in job_superstructure_list
    }
    return job_superstructure_dict


def sequential_extract_superstructure(preprocessed_job_data, change_points):
    job_superstructure_dict = {}
    for sensor, df in preprocessed_job_data.items():
        job_superstructure_dict[sensor] = extraction.detect_superstructure(
            df, change_points=list(change_points)
        )
    return job_superstructure_dict


@command()
def main(job_id, string_only=('s', False, 'Save only string dict')):
    """ This script does the following:
            1. Load job_data dict from pickle  (data/labeled_node_average_data/job<job_id>.pickle)
            2. Preprocess data
            3. Detect change points
            4. Extract structural features
            5. Save it to pickle file to (data/extracted/job<job_id>.pickle)
            """
    final_job_dict = {}

    node_average_data_file_path = '../../data/labeled_node_average_data/job{}.pickle'.format(job_id)
    with open(node_average_data_file_path, 'rb') as pickle_file:
        averaged_job_data = pickle.load(pickle_file)
    for sensor in averaged_job_data:
        averaged_job_data[sensor].name = sensor
    print('Lengths of each sensor time-series:')
    [print(len(a), end=' ') for a in averaged_job_data.values()]
    print()

    preprocessed_job_data, time_delta = preprocessing.preprocess_raw_sensor_data(averaged_job_data)
    print('Lengths of preprocessed time-series:')
    [print(len(a), end=' ') for a in preprocessed_job_data.values()]
    print()
    final_job_dict['preprocessed_job_data'] = preprocessed_job_data

    elapsed = time.time()
    change_points, change_points_t = cpd.detect_change_points(preprocessed_job_data, time_delta)
    print(
        'Change points detected: {}, time elapsed: {}'.format(change_points, time.time() - elapsed)
    )
    final_job_dict['change_points'] = change_points
    final_job_dict['change_points_t'] = change_points_t

    elapsed = time.time()
    job_superstructure_dict = parallel_extract_superstructure(preprocessed_job_data, change_points)
    # job_superstructure_dict = sequential_extract_superstructure(preprocessed_job_data, change_points)
    print('Time elapsed on feature extraction: {}'.format(time.time() - elapsed))
    final_job_dict['superstructure_dict'] = job_superstructure_dict

    encoded_structure_dict = {}
    for sensor, superstructure in job_superstructure_dict.items():
        encoded_structure_dict[sensor] = job_superstructure_dict[sensor].get_string()
    final_job_dict['encoded_structure_dict'] = encoded_structure_dict

    if string_only:
        string_data_file_path = '../../data/string_extracted/job{}.pickle'.format(job_id)
        with open(string_data_file_path, 'wb+') as pickle_file:
            pickle.dump(encoded_structure_dict, pickle_file)

    full_final_data_file_path = '../../data/extracted/job{}.pickle'.format(job_id)
    with open(full_final_data_file_path, 'wb+') as pickle_file:
        pickle.dump(final_job_dict, pickle_file)


if __name__ == '__main__':
    main.command()
