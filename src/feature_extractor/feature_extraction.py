from collections import namedtuple
import pandas as pd
import numpy as np

from . import structures, change_point_detection


class SuperstructureApproximation:
    def __init__(self):
        self.sum_error = 0
        self.partitioning = []
        self.structures = []
        self.errors = []

    def __repr__(self):
        return str(self.__dict__)

    def get_string(self):
        return ''.join([structure.symbol for structure in self.structures])


def detect_optimal_structure(time_series):
    min_error = float('inf')
    min_structure = None
    for structure_class in structures.AllStructures:
        structure = structure_class.fit(time_series)
        if structure.error < min_error:
            min_error = structure.error
            min_structure = structure
    return min_structure


def detect_superstructure(time_series, partitioning=None, change_points=None, time_delta=None):
    superstructure = SuperstructureApproximation()
    superstructure.partitioning = partitioning or get_partitioning(
        time_series, change_points, time_delta
    )
    for left, right in superstructure.partitioning:
        subregion = time_series[left:right]
        subregion_structure = detect_optimal_structure(subregion)
        superstructure.structures.append(subregion_structure)
        superstructure.errors.append(subregion_structure.error)
        superstructure.sum_error += subregion_structure.error
    return superstructure


def get_partitioning(time_series, change_points=None, time_delta=None):
    if not change_points:
        change_points, change_points_times = change_point_detection.detect_change_points(
            [time_series], time_delta
        )
    return [(change_points[i], change_points[i + 1]) for i in range(len(change_points) - 1)]


def superstructure_to_feature_vector(superstructure):
    feature_vector = np.array([])
    for structure in superstructure.structures:
        structure_features = structure.get_feature_vector()
        feature_vector = np.append(feature_vector, structure_features)
    return feature_vector


def extract_features(time_series):
    return superstructure_to_feature_vector(detect_superstructure(time_series))
