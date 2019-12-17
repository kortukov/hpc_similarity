from collections import namedtuple
import pandas as pd
import numpy as np

from . import structures


class SuperstructureApproximation:
    def __init__(self):
        self.sum_error = 0
        self.partitioning = []
        self.structures = []
        self.errors = []

    def __repr__(self):
        return str(self.__dict__)


class FeatureExtractor:
    def __init__(self, number_of_subregions=10):
        self.number_of_subregions = number_of_subregions

    @staticmethod
    def detect_optimal_structure(time_series):
        min_error = float('inf')
        min_structure = None
        for structure_class in structures.AllStructures:
            structure = structure_class.fit(time_series)
            if structure.error < min_error:
                min_error = structure.error
                min_structure = structure
        return min_structure

    @staticmethod
    def detect_superstructure(time_series, number_of_subregions=10):
        superstructure = SuperstructureApproximation()
        superstructure.partitioning = FeatureExtractor.get_partitioning(time_series, number_of_subregions)
        for left, right in superstructure.partitioning:
            subregion = time_series[left:right]
            subregion_structure = FeatureExtractor.detect_optimal_structure(subregion)
            superstructure.structures.append(subregion_structure)
            superstructure.errors.append(subregion_structure.error)
            superstructure.sum_error += subregion_structure.error
        return superstructure

    @staticmethod
    def get_partitioning(time_series, number_of_subregions):
        """
        Function
        :param time_series: pandas dataframe with columns 't' and 'y'
        :param number_of_subregions: int
        :return:
            list of 2-tuples - [(x1,x2), (x3,x4), ... ] partitioning of the time
            series into subregions
        """
        length = time_series['t'].count()
        if length < number_of_subregions * 2:
            raise ValueError('Too small time_series')

        delta = length // number_of_subregions
        left = 0
        partitioning = []
        while left < length:
            right = min(left + delta, length)
            partitioning.append((left, right))
            left = right + 1

        return partitioning

    @staticmethod
    def superstructure_to_feature_vector(superstructure):
        feature_vector = np.array([])
        for structure in superstructure.structures:
            structure_features = structure.get_feature_vector()
            feature_vector = np.append(feature_vector, structure_features)
        return feature_vector
