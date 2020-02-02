import numpy as np
from numba import jit
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

    change_points = change_points[1:-1]
    change_points = postfilter(multivariate_ts, change_points)

    change_points = np.append(change_points, [1, len(multivariate_ts)])
    change_points = np.unique(change_points)

    change_points_times = [list(time_series_dict.values())[0].t[ind] for ind in change_points]
    return list(change_points), change_points_times


def postfilter(X, change_points):
    if len(change_points) > 0:
        change_points = detect_local_lines(X, change_points, np.pi / 75, 5)
        # change_points = split_30(X, change_points)
    return change_points


# выделение окрестностей
def detect_local_lines(Z, change_points, threshold_angle, radius):
    X = Z * 5
    n = len(X)
    char_n = len(X[0])
    new_ch_points = []
    res_angles = []

    half_radius = int(radius / 2)

    for change_point in change_points:
        left = change_point - 10
        right = change_point + 10
        if left < 0:
            left = 0
        if right >= n:
            right = n - 1

        coef_l = np.ones((char_n))
        coef_r = np.ones((char_n))
        angles = []
        ang_points = []
        left_point = change_point
        right_point = change_point

        tmp = np.zeros((char_n))

        for point in range(left, change_point):
            for j in range(char_n):
                if point >= half_radius:
                    l = point - radius
                    if l < 0:
                        l = 0

                    weights = np.zeros((point + 1 - l))
                    weights[-1] = 1
                    weights[:-1] = 0.1

                    coef = np.polynomial.polynomial.polyfit(
                        range(l, point + 1), X[l : point + 1, j], 1, w=weights
                    )
                    coef_l[j] = coef[1]
                if point < n - 3:
                    r = point + 5
                    if r >= n:
                        r = n - 1

                    weights = np.zeros((r + 1 - point))
                    weights[0] = 1
                    weights[1:] = 0.1

                    coef = np.polynomial.polynomial.polyfit(
                        range(point, r + 1), X[point : r + 1, j], 1, w=weights
                    )
                    coef_r[j] = coef[1]
            coef_l = np.arctan(coef_l)
            coef_r = np.arctan(coef_r)
            for j in range(char_n):
                angle = abs(coef_l[j] - coef_r[j])
                if angle > np.pi / 2:
                    angle = np.pi - angle
                tmp[j] = angle
            angles.append(np.average(tmp))
            ang_points.append(point)
        zipped = zip(angles, ang_points)
        zipped = sorted(zipped, reverse=True)
        # print(zipped)
        l_angle = 0
        # print(zipped[0])
        if zipped[0][0] > threshold_angle:
            left_point = zipped[0][1]
            l_angle = zipped[0][0]

        coef_l[:] = 1
        coef_r[:] = 1
        angles = []
        ang_points = []
        tmp = np.zeros((char_n))
        for point in range(change_point, right + 1):
            for j in range(char_n):
                if point >= half_radius:
                    l = point - radius
                    if l < 0:
                        l = 0
                    coef = np.polynomial.polynomial.polyfit(
                        range(l, point + 1), X[l : point + 1, j], 1
                    )
                    coef_l[j] = coef[1]
                if point < n - half_radius - 1:
                    r = point + radius
                    if r >= n:
                        r = n - 1
                    coef = np.polynomial.polynomial.polyfit(
                        range(point, r + 1), X[point : r + 1, j], 1
                    )
                    coef_r[j] = coef[1]
            coef_l = np.arctan(coef_l)
            coef_r = np.arctan(coef_r)

            for j in range(char_n):
                angle = abs(coef_l[j] - coef_r[j])
                if angle > np.pi / 2:
                    angle = np.pi - angle
                tmp[j] = angle
            angles.append(np.average(tmp))
            ang_points.append(point)

        zipped = zip(angles, ang_points)
        zipped = sorted(zipped, reverse=True)
        # print(zipped)
        r_angle = 0
        # print(zipped[0])
        # print("")
        if zipped[0][0] > threshold_angle:
            right_point = zipped[0][1]
            r_angle = zipped[0][0]

        if left_point == right_point:
            new_ch_points.append(left_point)
            res_angles.append(l_angle)
        else:
            if left_point not in new_ch_points:
                new_ch_points.append(left_point)
                res_angles.append(l_angle)
            if right_point not in new_ch_points:
                new_ch_points.append(right_point)
                res_angles.append(r_angle)
    # print(new_ch_points)
    zipped = zip(new_ch_points, res_angles)
    zipped = sorted(zipped)
    new_ch_points = [x[0] for x in zipped]
    res_angles = [x[1] for x in zipped]
    new_ch_points = remove_close(new_ch_points, res_angles)

    # print(new_ch_points)
    return new_ch_points


# удаление точек, которые находятся очень близко друг от друга
def remove_close(change_points, angles):
    import sys

    old_one = np.copy(change_points)
    found = False
    k = 0
    while True:
        found = False
        new_ch_points = []
        cur = 0
        if len(change_points) == 1:
            new_ch_points = change_points
            break
        for i in range(1, len(change_points) - 1):
            if change_points[i] == change_points[cur]:
                # print(str(i) + " " + str(cur))
                found = True
                continue
            if change_points[i] - change_points[cur] <= 2:
                found = True
                # print("cur:{}, i={}".format(change_points[cur], change_points[i]))
                if angles[i] > angles[cur]:
                    new_ch_points.append(change_points[i])
                    cur = i
                else:
                    new_ch_points.append(change_points[cur])
            else:
                new_ch_points.append(change_points[cur])
                cur = i
        if (
            len(new_ch_points) > 0
            and len(change_points) > 0
            and new_ch_points[-1] != change_points[cur]
        ):
            new_ch_points.append(change_points[cur])
        new_ch_points.append(old_one[-1])
        new_ch_points = np.unique(new_ch_points)
        if not found:
            break
        change_points = np.copy(new_ch_points)

    k = 1
    while True:
        if k > len(new_ch_points):
            break
        if old_one[-1] - new_ch_points[-k] <= 2:
            k += 1
        else:
            break
    k -= 1
    if k > 0:
        new_ch_points = new_ch_points[:-k]

    new_ch_points = np.unique(np.append(new_ch_points, old_one[-1]))
    return new_ch_points


def split_30(X, change_points):
    n = len(X)
    new_ch_points = []
    change_points.append(0)
    change_points.append(n - 1)
    change_points.sort()
    for i in range(len(change_points) - 1):
        new_ch_points.append(change_points[i])
        if change_points[i + 1] - change_points[i] > 8:
            for j in range(change_points[i] + 6, change_points[i + 1], 6):
                new_ch_points.append(j)

    new_ch_points.sort()
    new_ch_points = new_ch_points[1:]
    return new_ch_points
