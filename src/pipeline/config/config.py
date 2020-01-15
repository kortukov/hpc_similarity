import json
CONFIG_PATH = '/Users/eekortukov/Parallel/hpc_similarity/src/pipeline/config/hpc_similarity.conf'

try:

    with open(CONFIG_PATH, 'r') as config_file:
        config = json.load(config_file)

except Exception as e:
    raise ValueError('Failed to load config file %s: %s' % (CONFIG_PATH, e))

CONNECTION_STRING = config['connection_string']
SENSORS_LIST = config['sensors_list']