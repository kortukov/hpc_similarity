import datetime
from opster import command
import psycopg2
import psycopg2.extras
from pprint import pprint

import config.config as config


@command()
def main(length=('l', 0, 'Length of job in minutes')):
    """Show last 10 finished jobs. Length is specified with parameter"""
    db = psycopg2.connect(config.CONNECTION_STRING)
    cursor = db.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    needed_fields = ('job_id', 't_start', 't_end')
    if not length:
        query = 'SELECT {}, {}, {} FROM job WHERE t_end < {} ORDER BY t_end DESC LIMIT 10'.format(
            *needed_fields, int(datetime.datetime.now().timestamp())
        )
    else:
        query = (
            'SELECT {}, {}, {} FROM job WHERE t_end < {} AND t_end - t_start >= {} '
            'AND t_end - t_start <= {} ORDER BY t_end DESC LIMIT 10'.format(
                *needed_fields,
                int(datetime.datetime.now().timestamp()),
                length * 60 - 600,
                length * 60 + 600,
            )
        )
    cursor.execute(query)
    nodes_string = None
    job_data_list = []
    for row in cursor:
        job_data = {}
        for key in row:
            job_data[key] = row[key]
            if key in ('t_start', 't_end'):
                job_data[key + '_utc'] = str(datetime.datetime.utcfromtimestamp(row[key]))
        job_data['minutes'] = (job_data['t_end'] - job_data['t_start']) / 60
        job_data_list.append(job_data)

    for job_data in job_data_list:
        if job_data:
            pprint(job_data)
    if not nodes_string:
        exit('No such job')


if __name__ == '__main__':
    main.command()
