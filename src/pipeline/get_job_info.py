import datetime
from opster import command
from pprint import pprint
import psycopg2
import psycopg2.extras

import config.config as config


@command()
def main(job_id):
    """Get info on job from job_data_db"""
    db = psycopg2.connect(
        config.CONNECTION_STRING
    )
    cursor = db.cursor(cursor_factory = psycopg2.extras.RealDictCursor)
    query = 'SELECT * FROM job WHERE job_id = {}'.format(job_id)
    cursor.execute(query)
    job_data = {}
    for row in cursor:
        for key in row:
            job_data[key] = row[key]
            if key in ('t_start', 't_end'):
                job_data[key+'_utc'] = str(datetime.datetime.utcfromtimestamp(row[key]))
        break
    if job_data:
        pprint(job_data)
    else:
        print('No job data found')


if __name__ == '__main__':
    main.command()
