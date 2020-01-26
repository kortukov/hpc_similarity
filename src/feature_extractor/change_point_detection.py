import numpy as np

from rpy2.robjects.packages import importr

e = importr('ecp')
from rpy2.robjects import numpy2ri

numpy2ri.activate()


def detect_change_points(time_series_dict, time_delta):
    """Uses 'ecp' R-package to detect change points in multivariate time-series
    Returns 2-tuple of lists. First element is indexes of change points, including first and
        last index in time-series. Second element is timestamps of changepoints, for plotting.

    """
    min_size = int(30 / time_delta) + 1

    multivariate_ts = np.stack([np.array(ts.y) for ts in time_series_dict.values()])
    multivariate_ts = multivariate_ts.transpose()

    estimated = e.e_divisive(multivariate_ts, sig_lvl=0.1, min_size=min_size)
    change_points = np.array(estimated[estimated.names.index('estimates')], dtype=np.int64)
    change_points[-1] -= 1
    change_points_times = [list(time_series_dict.values())[0].t[ind] for ind in change_points]
    return change_points, change_points_times
