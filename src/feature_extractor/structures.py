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

    @classmethod
    def fit(cls, time_series):
        """Fit structure to provided time series.
        Args:
            time_series (pandas.DataFrame): must have 2 columns 't' and 'y'.
        """
        parameters = cls._compute_parameters(time_series)
        return cls(parameters, time_series['t'])

    @classmethod
    def _compute_parameters(cls, time_series):
        raise NotImplementedError

    def get_y_series(self):
        raise NotImplementedError

    def get_df(self):
        y_series = self.get_y_series()
        return pd.DataFrame({'t': self.t_series, 'y': y_series})


class ConstantStructure(Structure):
    """ f(t) = a"""

    @classmethod
    def _compute_parameters(cls, time_series):
        return ConstantParameters(time_series['y'].mean())

    def get_y_series(self):
        return pd.Series([self.parameters.a] * self.t_series.size)


class StraightStructure(Structure):
    """ f(t) = a + b*t"""

    @classmethod
    def _compute_parameters(cls, time_series):
        t_series = time_series['t']
        y_series = time_series['y']
        t_mean = t_series.mean()
        y_mean = y_series.mean()
        b = sum((t_series - t_mean) * (y_series - y_mean)) / sum((t_series - t_mean) ** 2)
        a = y_mean - b * t_mean
        return StraightParameters(a, b)

    def get_y_series(self):
        return self.t_series * self.parameters.b + self.parameters.a


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
        print(res)
        return ExponentialParameters(*res.x)

    def get_y_series(self):
        return self.parameters.a * abs(self.parameters.b) ** self.t_series + self.parameters.c


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
        print(res)
        return SinusoidalParameters(*res.x)

    def get_y_series(self):
        return self.parameters.a * np.sin(self.t_series + self.parameters.b) + self.parameters.c


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
        print(res)
        return TriangularParameters(*res.x)

    def get_y_series(self):
        """Note: logical indexing is the best way to implement piecewise funtions to pd.Series i could find."""
        return (
            self.parameters.a + self.t_series[self.t_series < self.parameters.c] * self.parameters.b
        ).append(
            (self.parameters.a + 2 * self.parameters.b * self.parameters.c)
            - (self.parameters.b * self.t_series[self.t_series >= self.parameters.c])
        )


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
        print(res)
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
