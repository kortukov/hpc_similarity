## hpc_similarity
A tool to find similar supercomputer applications using system monitoring time-series data and structural pattern recognition.


####Workflow
1. Use R ecp package to divide sensor time-series into intervals.
https://cran.r-project.org/web/packages/ecp/index.html
2. Extract structural features from each subinterval. Idea from Generalized Feature Extraction 
for Structural Pattern Recognition in Time-Series Data Robert T. Olszewski February 2001
https://www.cs.cmu.edu/~bobski/pubs/tr01108-twosided.pdf
3. Encode extracted structural primitives with letters. 
4. Use string distance metrics to tell distance between two supercomputer jobs.
5. Train a classifier to discriminate between similar and not similar pairs of jobs.


####Code structure
`src/pipeline` - *Workflow scripts*:
* `load_raw_job_from_db.py` - Load sensor data from database.
* `extract_features.py` - Preprocess data, find intervals with ecp, extract structural features.
* `classify.py` - Using encoded jobs and labeled data, calculate string distances, classify jobs and measure accuracy.

Also some useful additional tools:
* `get_job_info.py` - Get job info from db for specified job_id.
* `show_last_jobs.py` - Show 10 last jobs of specified length.
* `read_job_string_dict.py` - Show extracted string features for specified job_id.

Scripts use a config file:
* `config/config.py` - Reads the JSON config to python.
* `config/hpc_similarity.conf` - JSON config file.


`src/feature_extractor` - *Feature extraction library*:
* `preprocessing.py` - All preprocessing done to sensor time series.
* `structures.py` - Classes for all structural primitives.
* `feature_extraction.py` - Extraction of optimal structural features.
* `change_point_detection.py` - Finding change points with ecp.
* `plotting.py` - Useful plotting functions.


####Data structure

`data/labeled_node_average_data` - Job data downloaded from db. For every job - one pickle file 'job{job_id}.pickle'

`data/train_data` - Contains `train.json` and `test.json` - JSON files with labeled pairs looking like this:
`{'job1:job2': true, 'job2:job3':false, 'job1:job3': true}`
These are train and test datasets respectively.

`data/extracted` - Feature extraction results for every job. For every job - one pickle file 'job{job_id}.pickle.
It contains fields `'preprocessed_job_data', 'change_points', 'change_points_t', 'superstructure_dict', 'encoded_structure_dict'`.

`data/string_extracted` - Only extracted dict with extracted strings for every sensor. One pickle file for every job.