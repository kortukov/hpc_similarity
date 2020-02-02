from opster import command
import pickle
import pprint


@command()
def main(job_id):
    string_data_file_path = '../../data/string_extracted/job{}.pickle'.format(job_id)
    with open(string_data_file_path, 'rb') as pickle_file:
        string_dict = pickle.load(pickle_file)
    string_dict['job_id'] = job_id
    pprint.pprint(string_dict)


if __name__ == '__main__':
    main.command()
