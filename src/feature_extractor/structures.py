from collections import namedtuple
import pandas as pd
import numpy as np
from scipy.optimize import minimize, curve_fit

ConstantParameters = namedtuple('ConstantParameters', 'a')
StraightParameters = namedtuple('StraightParameters', 'a b')
ExponentialParameters = namedtuple('ExponentialParameters', 'a b c')
SinusoidalParameters = namedtuple('SinusoidalParameters', 'a b c')
TriangularParameters = namedtuple('TriangularParameters', 'a b c')
TrapezoidalParameters = namedtuple('TrapezoidalParameters', 'a b c')
ZFormParameters = namedtuple('ZFormParameters', 'a b c_start c_stop')
InverseZFormParameters = namedtuple('InverseZFormParameters', 'a b c_start c_stop')

DISPLAY_SEARCH = True


alternative_encoding_map = {
    'a': 'a',
    'b': 'b',
    'c': 'c',
    'd': 'd',
    'e': 'e',
    'f': 'f',
    'g': 'bc',
    'h': 'cb',
    'i': 'bac',
    'j': 'cab',
    'k': 'aba',
    'l': 'aca',
    'm': 'bab',
    'n': 'cac',
}


class Structure:
    """Base class for all 6 structures
    Args:
        parameters (obj of *Parameters): namedtuple of appropriate class.
        t_series (pandas.Series)

    Note: Class instances are created by providing time_series to Structure.fit() method.
    """

    def __init__(self, parameters, t_series):
        self.parameters = parameters
        self.t_series = t_series
        self.error = None

    @classmethod
    def fit(cls, time_series):
        """Fit structure to provided time series.
        Args:
            time_series (pandas.DataFrame): must have 2 columns 't' and 'y'.
        """
        parameters = cls._compute_parameters(time_series)
        fitted_structure = cls(parameters, time_series['t'])
        fitted_structure.calculate_error(time_series)
        return fitted_structure

    @classmethod
    def _compute_parameters(cls, time_series):
        raise NotImplementedError

    def get_y_series(self):
        raise NotImplementedError

    def calculate_error(self, time_series):
        self.error = sum((self.y_series - time_series['y']) ** 2)

    def get_df(self):
        y_series = self.get_y_series()
        return pd.DataFrame({'t': self.t_series, 'y': y_series})

    def get_feature_vector(self):
        raise NotImplementedError

    @property
    def y_series(self):
        return self.get_y_series()

    @property
    def symbol(self):
        raise NotImplementedError

    @property
    def alternative_symbol(self):
        return alternative_encoding_map[self.symbol]

    def __repr__(self):
        return str(type(self))


class ConstantStructure(Structure):
    """ f(t) = a"""

    @classmethod
    def _compute_parameters(cls, time_series):
        return ConstantParameters(time_series['y'].mean())

    def get_y_series(self):
        y_series = pd.Series([self.parameters.a] * self.t_series.size)
        y_series.index = self.t_series.index
        return y_series

    def get_feature_vector(self):
        feature_vector = [0.0] * 15
        feature_vector[0] = self.parameters.a
        return np.array(feature_vector)

    @property
    def symbol(self):
        return 'a'


class StraightStructure(Structure):
    """ f(t) = a + b*t"""

    @classmethod
    def _compute_parameters(cls, time_series):
        t_series = time_series['t']
        y_series = time_series['y']
        t_mean = t_series.mean()
        y_mean = y_series.mean()
        b = sum((t_series - t_mean) * (y_series - y_mean)) / (sum((t_series - t_mean) ** 2) or 1)
        a = y_mean - b * t_mean
        return StraightParameters(a, b)

    def get_y_series(self):
        return self.t_series * self.parameters.b + self.parameters.a

    def get_feature_vector(self):
        feature_vector = [0.0] * 15
        feature_vector[1] = self.parameters.a
        feature_vector[2] = self.parameters.b
        return np.array(feature_vector)

    @property
    def symbol(self):
        if self.parameters.b < 0:
            return 'b'
        elif self.parameters.b > 0:
            return 'c'
        else:
            return 'a'


class ExponentialStructure(Structure):
    """ f(t) = a * |b|**t + c"""

    @classmethod
    def _compute_parameters(cls, time_series):
        t_series = time_series['t']
        y_series = time_series['y']

        if len(y_series) < 3:
            return ExponentialParameters(0, 0, y_series.mean())
        try:
            params, cov = curve_fit(
                cls.exponential, t_series, y_series, p0=(1, 2, y_series.mean()), maxfev=2000
            )
        except RuntimeError:
            try:
                params, cov = curve_fit(
                    cls.exponential, t_series, y_series, p0=(-1, 2, y_series.mean()), maxfev=4000
                )
            except RuntimeError:
                return ExponentialParameters(0, 0, y_series.mean())
        return ExponentialParameters(*params)

    @staticmethod
    def exponential(t_series, a, b, c):
        t_series = (t_series - t_series.mean()) / t_series.std()
        return a * abs(b) ** t_series + c

    def get_y_series(self):
        return self.exponential(
            self.t_series, self.parameters.a, self.parameters.b, self.parameters.c
        )

    def get_feature_vector(self):
        feature_vector = [0.0] * 15
        feature_vector[3] = self.parameters.a
        feature_vector[4] = self.parameters.b
        feature_vector[5] = self.parameters.c
        return np.array(feature_vector)

    @property
    def symbol(self):
        if self.parameters.a == 0 or self.parameters.b == 0:
            # as const
            return 'a'
        if 0.98 < self.parameters.b < 1.02:
            # as straight
            if self.parameters.a < 0:
                return 'b'
            else:
                return 'c'
        if -1 < self.parameters.b < 1:
            return 'd'
        else:
            return 'e'


class SinusoidalStructure(Structure):
    """ f(t) = a * sin(t + b) + c"""

    @classmethod
    def _compute_parameters(cls, time_series):
        t_series = time_series['t']
        y_series = time_series['y']

        def minimized_function(params):
            parameters = SinusoidalParameters(*params)
            structure = SinusoidalStructure(parameters, t_series)
            struct_y_series = structure.get_y_series()
            return sum((y_series - struct_y_series) ** 2)

        x0 = np.array([0, 0, 0])
        res = minimize(minimized_function, x0, method='Nelder-Mead')
        return SinusoidalParameters(*res.x)

    def get_y_series(self):
        return self.parameters.a * np.sin(self.t_series + self.parameters.b) + self.parameters.c

    def get_feature_vector(self):
        feature_vector = [0.0] * 15
        feature_vector[6] = self.parameters.a
        feature_vector[7] = self.parameters.b
        feature_vector[8] = self.parameters.c
        return np.array(feature_vector)

    @property
    def symbol(self):
        if self.parameters.a == 0:
            # as const
            return 'a'
        return 'f'


class TriangularStructure(Structure):
    """ f(t) =
        a + b*t if t < c
        (a + 2*b*c) - (b*t) if t >= c
    """

    @classmethod
    def _compute_parameters(cls, time_series):
        t_series = time_series['t']
        y_series = time_series['y']

        if len(y_series) < 3:
            return TriangularParameters(y_series.mean(), 0, t_series.mean())

        try:
            params, cov = curve_fit(
                cls.triangular,
                t_series,
                y_series,
                p0=(y_series.mean(), y_series.mean() / (t_series.mean() or 1), t_series.mean()),
                bounds=([-np.inf, -np.inf, t_series.min()], [np.inf, np.inf, t_series.max()]),
                maxfev=1200,
            )
        except RuntimeError:
            return TriangularParameters(y_series.mean(), 0, t_series.mean())

        return TriangularParameters(*params)

    @staticmethod
    def triangular(t_series, a, b, c):
        return (a + t_series[t_series < c] * b).append(
            (a + 2 * b * c) - (b * t_series[t_series >= c])
        )

    def get_y_series(self):
        """Note: logical indexing is the best way to implement piecewise funtions to pd.Series i could find."""
        return self.triangular(
            self.t_series, self.parameters.a, self.parameters.b, self.parameters.c
        )

    def get_feature_vector(self):
        feature_vector = [0.0] * 15
        feature_vector[9] = self.parameters.a
        feature_vector[10] = self.parameters.b
        feature_vector[11] = self.parameters.c
        return np.array(feature_vector)

    @property
    def symbol(self):
        if -0.001 < self.parameters.b < 0.001:
            # as const
            return 'a'
        if self.parameters.b < 0:
            return 'g'
        else:
            return 'h'


class TrapezoidalStructure(Structure):
    """ f(t) =
        a + b*t if t < c_start
        a + b*c_start if  c_start <= t <c_stop
        (a + b*c_start +b*c_stop)−(b*t) if t >= c_stop
    """

    @classmethod
    def _compute_parameters(cls, time_series):
        t_series = time_series['t']
        y_series = time_series['y']

        if len(y_series) < 3:
            return TrapezoidalParameters(y_series.mean(), 0, t_series.mean())
        try:
            params, cov = curve_fit(
                cls.trapezoidal,
                t_series,
                y_series,
                p0=(y_series.mean(), (y_series.max() - y_series.mean()), 0),
                bounds=(
                    [-np.inf, -np.inf, 0],
                    [np.inf, np.inf, (t_series.max() - t_series.min()) / t_series.std()],
                ),
                maxfev=2000,
            )
        except RuntimeError:
            return TrapezoidalParameters(y_series.mean(), 0, t_series.mean())

        return TrapezoidalParameters(*params)

    @staticmethod
    def trapezoidal(t_series, a, b, c):
        t_series = (t_series - t_series.min()) / t_series.std()
        t2 = t_series.max()
        t1 = t_series.min()
        c_start = t1 + (c - t1) / 2
        c_stop = t1 + t2 - c_start
        return (
            (a + t_series[t_series < c_start] * b).append(
                (a + b * c_start) + t_series[(c_start <= t_series) & (t_series < c_stop)] * 0
            )
        ).append((a + b * c_start + b * c_stop - t_series[t_series >= c_stop] * b))

    def get_y_series(self):
        return self.trapezoidal(
            self.t_series, self.parameters.a, self.parameters.b, self.parameters.c
        )

    def get_feature_vector(self):
        feature_vector = [0.0] * 15
        feature_vector[12] = self.parameters.a
        feature_vector[13] = self.parameters.b
        feature_vector[14] = self.parameters.c
        return np.array(feature_vector)

    @property
    def symbol(self):
        if -0.001 < self.parameters.b < 0.001:
            return 'a'
        if self.parameters.b < 0:
            return 'i'
        else:
            return 'j'


class ZFormStructure(Structure):
    """ f(t) =
        a + b*c_start if t < c_start
        a + b*t if c_start <= t < c_stop
        a + b*c_stop if t >= c_stop
    """

    @classmethod
    def _compute_parameters(cls, time_series):
        t_series = time_series['t']
        y_series = time_series['y']

        if len(y_series) < 3:
            return ZFormParameters(y_series.mean(), 0, t_series.mean(), t_series.mean())

        try:
            params, cov = curve_fit(
                cls.z_form,
                t_series,
                y_series,
                p0=(
                    y_series.mean(),
                    y_series.mean(),
                    0,
                    (t_series.max() - t_series.min()) / t_series.std(),
                ),
                bounds=(
                    [-np.inf, -np.inf, 0, ((t_series.max() - t_series.min()) / t_series.std()) / 2],
                    [
                        np.inf,
                        np.inf,
                        ((t_series.max() - t_series.min()) / t_series.std()) / 2,
                        ((t_series.max() - t_series.min()) / t_series.std()),
                    ],
                ),
                maxfev=2000,
            )
        except RuntimeError:
            return ZFormParameters(y_series.mean(), 0, t_series.mean(), t_series.mean())

        return ZFormParameters(*params)

    @staticmethod
    def z_form(t_series, a, b, c_start, c_stop):
        t_series = (t_series - t_series.min()) / t_series.std()
        return (
            ((a + b * c_start) + t_series[t_series < c_start] * 0).append(
                (a + b * t_series[(c_start <= t_series) & (t_series < c_stop)])
            )
        ).append(((a + b * c_stop) + t_series[t_series > c_stop] * 0))

    def get_y_series(self):
        return self.z_form(
            self.t_series,
            self.parameters.a,
            self.parameters.b,
            self.parameters.c_start,
            self.parameters.c_stop,
        )

    @property
    def symbol(self):
        if -0.001 < self.parameters.b < 0.001:
            return 'a'
        if self.parameters.b < 0:
            return 'k'
        else:
            return 'l'


class InverseZFormStructure(Structure):
    """ f(t) =
        a + b*t if t < c_start
        a + b*c_start if  c_start <= t <c_stop
        (a + b*c_start +b*c_stop)+(b*t) if t >= c_stop
    """

    @classmethod
    def _compute_parameters(cls, time_series):
        t_series = time_series['t']
        y_series = time_series['y']

        if len(y_series) < 3:
            return InverseZFormParameters(y_series.mean(), 0, t_series.mean(), t_series.mean())
        try:
            params, cov = curve_fit(
                cls.inverse_z_form,
                t_series,
                y_series,
                p0=(
                    y_series.mean(),
                    (y_series.max() - y_series.mean()),
                    0,
                    (t_series.max() - t_series.min()) / t_series.std(),
                ),
                bounds=(
                    [-np.inf, -np.inf, 0, ((t_series.max() - t_series.min()) / t_series.std()) / 2],
                    [
                        np.inf,
                        np.inf,
                        ((t_series.max() - t_series.min()) / t_series.std()) / 2,
                        ((t_series.max() - t_series.min()) / t_series.std()),
                    ],
                ),
                maxfev=2000,
            )
        except RuntimeError:
            return InverseZFormParameters(y_series.mean(), 0, t_series.mean(), t_series.mean())

        return InverseZFormParameters(*params)

    @staticmethod
    def inverse_z_form(t_series, a, b, c_start, c_stop):
        t_series = (t_series - t_series.min()) / t_series.std()
        return (
            (a + t_series[t_series < c_start] * b).append(
                (a + b * c_start) + t_series[(c_start <= t_series) & (t_series < c_stop)] * 0
            )
        ).append((a + b * c_start - b * c_stop + b * t_series[t_series >= c_stop]))

    def get_y_series(self):
        return self.inverse_z_form(
            self.t_series,
            self.parameters.a,
            self.parameters.b,
            self.parameters.c_start,
            self.parameters.c_stop,
        )

    def get_feature_vector(self):
        feature_vector = [0.0] * 15
        feature_vector[12] = self.parameters.a
        feature_vector[13] = self.parameters.b
        feature_vector[14] = self.parameters.c_start
        return np.array(feature_vector)

    @property
    def symbol(self):
        if -0.001 < self.parameters.b < 0.001:
            return 'a'
        if self.parameters.b < 0:
            return 'm'
        else:
            return 'n'


AllStructures = (
    ConstantStructure,
    StraightStructure,
    ExponentialStructure,
    SinusoidalStructure,
    TriangularStructure,
    TrapezoidalStructure,
    ZFormStructure,
    InverseZFormStructure,
)
