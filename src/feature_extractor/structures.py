from collections import namedtuple
import pandas as pd
import numpy as np
from scipy.optimize import minimize

ConstantParameters = namedtuple('ConstantParameters', 'a')
StraightParameters = namedtuple('StraightParameters', 'a b')
ExponentialParameters = namedtuple('ExponentialParameters', 'a b c')
SinusoidalParameters = namedtuple('SinusoidalParameters', 'a b c')
TriangularParameters = namedtuple('TriangularParameters', 'a b c')
TrapezoidalParameters = namedtuple('TrapezoidalParameters', 'a b c')


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
        self.error = sum((self.get_y_series() - time_series['y']) ** 2)

    def get_df(self):
        y_series = self.get_y_series()
        return pd.DataFrame({'t': self.t_series, 'y': y_series})

    def get_feature_vector(self):
        raise NotImplementedError

    @property
    def symbol(self):
        raise NotImplementedError

    def __repr__(self):
        return str(type(self))


class ConstantStructure(Structure):
    """ f(t) = a"""

    @classmethod
    def _compute_parameters(cls, time_series):
        return ConstantParameters(time_series['y'].mean())

    def get_y_series(self):
        return pd.Series([self.parameters.a] * self.t_series.size)

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

        def minimized_function(params):
            parameters = ExponentialParameters(*params)
            structure = ExponentialStructure(parameters, t_series)
            struct_y_series = structure.get_y_series()
            return sum((y_series - struct_y_series) ** 2)

        x0 = np.array([0, 0, 0])
        res = minimize(minimized_function, x0, method='Nelder-Mead')
        return ExponentialParameters(*res.x)

    def get_y_series(self):
        return self.parameters.a * abs(self.parameters.b) ** self.t_series + self.parameters.c

    def get_feature_vector(self):
        feature_vector = [0.0] * 15
        feature_vector[3] = self.parameters.a
        feature_vector[4] = self.parameters.b
        feature_vector[5] = self.parameters.c
        return np.array(feature_vector)

    @property
    def symbol(self):
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

        def minimized_function(params):
            parameters = TriangularParameters(*params)
            structure = TriangularStructure(parameters, t_series)
            struct_y_series = structure.get_y_series()
            return sum((y_series - struct_y_series) ** 2)

        x0 = np.array([0, 0, 0])
        res = minimize(minimized_function, x0, method='Nelder-Mead')
        return TriangularParameters(*res.x)

    def get_y_series(self):
        """Note: logical indexing is the best way to implement piecewise funtions to pd.Series i could find."""
        return (
            self.parameters.a + self.t_series[self.t_series < self.parameters.c] * self.parameters.b
        ).append(
            (self.parameters.a + 2 * self.parameters.b * self.parameters.c)
            - (self.parameters.b * self.t_series[self.t_series >= self.parameters.c])
        )

    def get_feature_vector(self):
        feature_vector = [0.0] * 15
        feature_vector[9] = self.parameters.a
        feature_vector[10] = self.parameters.b
        feature_vector[11] = self.parameters.c
        return np.array(feature_vector)

    @property
    def symbol(self):
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

        def minimized_function(params):
            parameters = TrapezoidalParameters(*params)
            structure = TrapezoidalStructure(parameters, t_series)
            struct_y_series = structure.get_y_series()
            return sum((y_series - struct_y_series) ** 2)

        x0 = np.array([0, 0, 0])
        res = minimize(minimized_function, x0, method='Nelder-Mead')
        return TrapezoidalParameters(*res.x)

    def get_y_series(self):
        # Note the & usage in middle part. Pandas needs this instead of python simple 'and'
        n = self.t_series.max()
        c_start = (n - self.parameters.c) / 2
        c_stop = n - c_start + 1
        return (
            (self.parameters.a + self.t_series[self.t_series < c_start] * self.parameters.b).append(
                (self.parameters.a + self.parameters.b * c_start)
                + self.t_series[(c_start <= self.t_series) & (self.t_series < c_stop)] * 0
            )
        ).append(
            (
                self.parameters.a
                + self.parameters.b * c_start
                + self.parameters.b * c_stop
                - self.t_series[self.t_series >= c_stop] * self.parameters.b
            )
        )

    def get_feature_vector(self):
        feature_vector = [0.0] * 15
        feature_vector[12] = self.parameters.a
        feature_vector[13] = self.parameters.b
        feature_vector[14] = self.parameters.c
        return np.array(feature_vector)

    @property
    def symbol(self):
        if self.parameters.b < 0:
            return 'i'
        else:
            return 'j'


AllStructures = (
    ConstantStructure,
    StraightStructure,
    ExponentialStructure,
    SinusoidalStructure,
    TriangularStructure,
    TrapezoidalStructure,
)
