import numpy as np

from rpy2.robjects.packages import importr
e = importr('ecp')
from rpy2.robjects import numpy2ri
numpy2ri.activate()


def detect_change_points(time_series_list):
    """Uses 'ecp' R-package to detect change points in multivariate time-series
    Returns 2-tuple of lists. First element is indexes of change points, including first and
        last index in time-series. Second element is timestamps of changepoints, for plotting.

    """
    multivariate_ts = np.stack([np.array(ts.y) for ts in time_series_list])
    multivariate_ts = multivariate_ts.transpose()

    estimated = e.e_divisive(multivariate_ts, sig_lvl=0.1, min_size=4)
    change_points = np.array(estimated[estimated.names.index('estimates')], dtype=np.int64)
    change_points[-1] -= 1
    change_points_times = [time_series_list[0].t[ind] for ind in change_points]
    return change_points, change_points_times

