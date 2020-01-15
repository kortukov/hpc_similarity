from opster import command
from pathlib import Path
import psycopg2
import psycopg2.extras

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


def download_job_data(cursor, job_id, nodes_list, t_start, t_end, sensor_tables=config.SENSORS_LIST):
    raw_job_data_dir_path = '../../data/raw_data/job{}'.format(job_id)
    Path(raw_job_data_dir_path).mkdir(parents=True, exist_ok=True)
    for sensor in sensor_tables:
        for node in nodes_list:
            csv_relative_path = '/{}{}.csv'.format(sensor, node)
            csv_absolute_path = '{}{}'.format(raw_job_data_dir_path, csv_relative_path)
            print(csv_absolute_path)
            query = (
                'SELECT time, avg '
                'FROM {sensor} '
                'WHERE node_id = {node} AND time >= {t_start} AND time <= {t_end}'.format(
                    sensor=sensor, node=node, t_start=t_start, t_end=t_end
                )
            )
            outputquery = "COPY ({}) TO STDOUT WITH CSV HEADER".format(query)
            with open(csv_absolute_path, 'w+') as f:
                cursor.copy_expert(outputquery, f)


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
