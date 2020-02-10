from opster import command
import psycopg2
import psycopg2.extras
import pickle
import pandas as pd

import config.config as config


def parse_nodes_string(nodes_string):
    for char in ('n', '[', ']'):
        nodes_string = nodes_string.replace(char, '')

    nodelist = []
    for nodes in nodes_string.split(','):
        if '-' in nodes:
            first_node, last_node = nodes.split('-')
            node_range = list(range(int(first_node), int(last_node) + 1))
            nodelist.extend(node_range)
        else:
            nodelist.append(int(nodes))
    return nodelist


def download_job_data(
    cursor, job_id, nodes_list, t_start, t_end, sensor_tables=config.SENSORS_LIST
):
    averaged_job_data = {}
    for sensor in sensor_tables:
        print('Downloading {} for job {}'.format(sensor, job_id))
        node_query_part = ",".join([str(x) for x in nodes_list])
        query = (
            'SELECT time, avg(avg) '
            'FROM {sensor} '
            'WHERE node_id in ({nodes}) AND time >= {t_start} AND time <= {t_end} group by time order by time '.format(
                sensor=sensor, nodes=node_query_part, t_start=t_start, t_end=t_end
            )
        )
        # outputquery = "COPY ({}) TO STDOUT WITH CSV HEADER".format(query)
        # with open(csv_absolute_path, 'w+') as f:
        #     cursor.copy_expert(outputquery, f)
        cursor.execute(query)
        data = cursor.fetchall()

        if not data:
            exit('Failed to fetch data')
        if len(data) > 1:
            data = data[1:]

        sensor_data = {'time': [x['time'] for x in data], 'avg': [x['avg'] for x in data]}
        averaged_job_data[sensor] = pd.DataFrame(data=sensor_data)

    node_average_data_file_path = '../../data/labeled_node_average_data/job{}.pickle'.format(job_id)
    with open(node_average_data_file_path, 'wb+') as pickle_file:
        pickle.dump(averaged_job_data, pickle_file)


@command()
def main(job_id):
    """Load raw job data from job_data_db and save it as many csv's (one per node and sensor)
        to raw_data/job<job_id>/ directory.
        """
    db = psycopg2.connect(config.CONNECTION_STRING)
    cursor = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    needed_fields = ('t_start', 't_end', 'nodelist')
    query = 'SELECT {}, {}, {} FROM job WHERE job_id = {}'.format(*needed_fields, job_id)
    cursor.execute(query)
    nodes_string = None
    for row in cursor:
        t_start = row['t_start']
        t_end = row['t_end']
        nodes_string = row['nodelist']
        break

    if not nodes_string:
        exit('No such job')

    nodes_list = parse_nodes_string(nodes_string)
    download_job_data(cursor, job_id, nodes_list, t_start, t_end)


if __name__ == '__main__':
    main.command()
