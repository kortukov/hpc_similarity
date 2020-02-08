import json
from opster import command
from os import listdir
from os.path import isfile, join
import pickle
from sklearn.ensemble import RandomForestClassifier
import textdistance

import config.config as config


def load_all_jobs():
    all_jobs_directory_path = '/Users/eekortukov/Parallel/hpc_similarity/data/string_extracted/'
    all_jobs_paths = [
        f for f in listdir(all_jobs_directory_path) if isfile(join(all_jobs_directory_path, f))
    ]
    all_jobs = {}
    for file_path in all_jobs_paths:
        job_id = file_path.rstrip('.pickle').split('job')[-1]
        with open(join(all_jobs_directory_path, file_path), 'rb') as file:
            job_dict = pickle.load(file)
            all_jobs[job_id] = job_dict

    return all_jobs


def load_train_data():
    train_data_path = '/Users/eekortukov/Parallel/hpc_similarity/data/train_data/train.json'
    # with open(train_data_path, 'rb') as file:
    # train_data = json.load(file)
    train_data = {'1077274:1077720': False, '1077274:1078502': False, '1077720:1078502': True}
    return train_data


def load_test_data():
    test_data_path = '/Users/eekortukov/Parallel/hpc_similarity/data/train_data/test.json'
    # with open(test_data_path, 'rb') as file:
    # test_data = json.load(file)
    test_data = {'1077274:1077720': False, '1077274:1078502': False, '1077720:1078502': True}
    return test_data


def calculate_distances(job1, job2):
    sensors = list(job1.keys())
    assert sensors == list(job2.keys())
    assert sensors == config.SENSORS_LIST
    distances = []
    for sensor in sensors:
        string1, string2 = job1[sensor], job2[sensor]
        # Here we can try various string distance methods to see which works better
        distance = textdistance.jaro_winkler(string1, string2)
        distances.append(distance)

    return distances


def get_distance_matrix(all_jobs, train_pairs):
    """Row is one pair of jobs. In it are 16 distances, for every sensor."""
    distance_matrix = []
    for job_id1, job_id2 in train_pairs:
        job1, job2 = all_jobs[job_id1], all_jobs[job_id2]
        distance_matrix.append(calculate_distances(job1, job2))

    return distance_matrix


def train_and_estimate_accuracy(all_jobs, train_data, test_data=None):
    train_pairs = [job_ids.split(':') for job_ids, label in train_data.items()]
    train_labels = [label for job_ids, label in train_data.items()]
    train_distance_matrix = get_distance_matrix(all_jobs, train_pairs)

    classifier = RandomForestClassifier(n_estimators=10)
    classifier.fit(train_distance_matrix, train_labels)
    print('Train accuracy: {}'.format(classifier.score(train_distance_matrix, train_labels)))

    if test_data:
        test_pairs = [job_ids.split(':') for job_ids, label in test_data.items()]
        test_labels = [label for job_ids, label in test_data.items()]
        test_distance_matrix = get_distance_matrix(all_jobs, test_pairs)
        print('Test accuracy: {}'.format(classifier.score(test_distance_matrix, test_labels)))

    model_path = '/Users/eekortukov/Parallel/hpc_similarity/src/pipeline/model.pickle'
    with open(model_path, 'wb+') as file:
        pickle.dump(classifier, file)


@command()
def main(test=('t', False, 'Estimate accuracy on test data')):
    all_jobs = load_all_jobs()
    train_data = load_train_data()
    test_data = load_test_data() if test else None
    train_and_estimate_accuracy(all_jobs, train_data, test_data)


if __name__ == '__main__':
    main.command()
