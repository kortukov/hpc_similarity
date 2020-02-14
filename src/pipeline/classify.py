import json
from opster import command
from os import listdir
from os.path import isfile, join
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import textdistance

import config.config as config


def load_all_jobs():
    all_jobs_directory_path = '/Users/eekortukov/Parallel/hpc_similarity/data/extracted/'
    all_jobs_paths = [
        f for f in listdir(all_jobs_directory_path) if isfile(join(all_jobs_directory_path, f))
    ]
    all_jobs = {}
    for file_path in all_jobs_paths:
        job_id = file_path.rstrip('.pickle').split('job')[-1]
        with open(join(all_jobs_directory_path, file_path), 'rb') as file:
            extracted_job_data = pickle.load(file)

        # This is where string encoding happens
        job_dict = {}
        for sensor, superstructure in extracted_job_data['superstructure_dict'].items():
            job_dict[sensor] = superstructure.get_string()
        all_jobs[job_id] = job_dict

    return all_jobs


def load_train_data():
    train_data_path = '/Users/eekortukov/Parallel/hpc_similarity/data/train_data/train.json'
    with open(train_data_path, 'rb') as file:
        train_data = json.load(file)
    return train_data


def load_test_data():
    test_data_path = '/Users/eekortukov/Parallel/hpc_similarity/data/train_data/test.json'
    with open(test_data_path, 'rb') as file:
        test_data = json.load(file)
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
        # distance = textdistance.levenshtein(string1, string2)
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
    print(
        'Random forest train accuracy: {}'.format(
            classifier.score(train_distance_matrix, train_labels)
        )
    )

    if test_data:
        test_pairs = [job_ids.split(':') for job_ids, label in test_data.items()]
        test_labels = [label for job_ids, label in test_data.items()]
        test_distance_matrix = get_distance_matrix(all_jobs, test_pairs)
        print(
            'Random forest test accuracy: {}'.format(
                classifier.score(test_distance_matrix, test_labels)
            )
        )

    model_path = '/Users/eekortukov/Parallel/hpc_similarity/src/pipeline/rf_model.pickle'
    with open(model_path, 'wb+') as file:
        pickle.dump(classifier, file)

    log_classifier = LogisticRegression(random_state=0, solver='liblinear')
    log_classifier.fit(train_distance_matrix, train_labels)
    print(
        'Logistic regression train accuracy: {}'.format(
            log_classifier.score(train_distance_matrix, train_labels)
        )
    )

    if test_data:
        test_pairs = [job_ids.split(':') for job_ids, label in test_data.items()]
        test_labels = [label for job_ids, label in test_data.items()]
        test_distance_matrix = get_distance_matrix(all_jobs, test_pairs)
        print(
            'Logistic regression test accuracy: {}'.format(
                log_classifier.score(test_distance_matrix, test_labels)
            )
        )

    log_reg_coefs = {
        sensor: coef for coef, sensor in zip(log_classifier.coef_[0], config.SENSORS_LIST)
    }
    print('Logistic regression coefs: {}')
    pprint.pprint(log_reg_coefs)

    model_path = '/Users/eekortukov/Parallel/hpc_similarity/src/pipeline/lr_model.pickle'
    with open(model_path, 'wb+') as file:
        pickle.dump(log_classifier, file)


@command()
def main(test=('t', False, 'Estimate accuracy on test data')):
    all_jobs = load_all_jobs()
    train_data = load_train_data()
    test_data = load_test_data() if test else None
    train_and_estimate_accuracy(all_jobs, train_data, test_data)


if __name__ == '__main__':
    main.command()
