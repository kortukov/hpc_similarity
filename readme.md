A tool to find similar HPC application using system monitoring time-series data.

Uses generalized structure feature extractor.

src/feature_extractor - lib to extract feature from job data
src/examples - jupyter notebooks that show how this works
src/pipeline - scripts for job data downloading, averaging, pickling and feature extraction

raw_data - contains folders with downloaded raw job data by node and sensor.
node_average_data - folders with job data averaged by node
pickled_job_data - for every job - one pickle file, inside is a list of dataframes (one for each sensor)