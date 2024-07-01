from typing import Any

from numpy_api import xp
import numpy as np

from SkillSim import SkillSim
from SkillGroup import SkillGroup
from SkillPopulation import SkillPopulation
from utils.MatrixSubsetIndexes import MatrixSubsetIndexes


class SkillSimCalculatorV3(SkillSim):
    _skill_group: SkillGroup
    _skill_population_matrix: Any

    _rca_matrix: Any
    _skill_sim_matrix: Any

    def __init__(self, skill_population: SkillPopulation | SkillGroup):
        self._skill_population = skill_population
        self._skill_population_matrix = xp.asarray(skill_population.matrix)
        self._rca_matrix = None
        self._skill_sim_matrix = None

    def calc_rca_matrix(self):
        num_skills_in_jobs = xp.sum(self._skill_population_matrix, axis=1)[
            :, xp.newaxis
        ]

        nums_jobs_with_skill = xp.sum(self._skill_population_matrix, axis=0)
        num_skills = xp.sum(self._skill_population_matrix)

        self._rca_matrix = (self._skill_population_matrix / num_skills_in_jobs) / (
            nums_jobs_with_skill / num_skills
        )

    def get_rca_matrix(self):
        return self._rca_matrix

    def set_rca_matrix(self, rca_matrix):
        self._rca_matrix = rca_matrix

    def calc_skill_sim_matrix(self):
        effective_use_matrix = xp.where(self._rca_matrix >= 1.0, 1.0, 0.0)

        effective_use_sum_matrix = xp.matmul(
            effective_use_matrix.T, effective_use_matrix
        )

        skill_count_vector = xp.diag(effective_use_sum_matrix)

        t1 = xp.tile(skill_count_vector, (len(skill_count_vector), 1))
        t2 = t1.T

        effective_use_sum_max = xp.maximum(t1, t2)

        self._skill_sim_matrix = effective_use_sum_matrix / effective_use_sum_max

    def get_skill_sim_matrix(self):
        return self._skill_sim_matrix

    def set_skill_sim_matrix(self, skill_sim_matrix):
        self._skill_sim_matrix = skill_sim_matrix

    def skill_set_one_hot_vector(self, matrix_subset: MatrixSubsetIndexes):
        # rename of previous calcv2 skill_set_vector method
        skill_vector = xp.clip(
            xp.sum(self._skill_population_matrix[matrix_subset.indexes], axis=0), None, 1
        )

        return skill_vector

    def skill_weight_vector(self, matrix_subset: MatrixSubsetIndexes):
        return xp.sum(
            self.skill_set_one_hot_vector(matrix_subset)
            * self._rca_matrix[matrix_subset.indexes],
            axis=0,
        ) / len(matrix_subset)

    def skill_set_similarity(
        self,
        matrix_subset_1: MatrixSubsetIndexes,
        matrix_subset_2: MatrixSubsetIndexes | None,
        skill_weight_vector = None,
    ) -> float:
        if self._rca_matrix is None:
            self.calc_rca_matrix()

        if self._skill_sim_matrix is None:
            self.calc_skill_sim_matrix()

        if matrix_subset_2 is None and skill_weight_vector is None:
            raise Exception(
                "A second matrix subset or custom skill weight vector needs to be provided. Both cannot be None."
            )

        # skills that aren't included in the skill set of subset 1 or 2 will have a weight of zero
        # when the element-wise dot product is calculated with the skill sim matrix this means
        # skills that weren't included in the subset will be zeroed out and not count towards the sum
        subset_1_weight_vector = self.skill_weight_vector(matrix_subset_1)
        subset_2_weight_vector = (
            self.skill_weight_vector(matrix_subset_2)
            if skill_weight_vector is None
            else skill_weight_vector
        )

        _, num_skills = self._skill_sim_matrix.shape
        skill_weight_matrix = xp.ones((num_skills, num_skills))
        skill_weight_matrix = (
            subset_1_weight_vector[:, xp.newaxis] * skill_weight_matrix
        )
        skill_weight_matrix = subset_2_weight_vector * skill_weight_matrix

        return np.float64(
            xp.sum(
                subset_2_weight_vector
                * (subset_1_weight_vector[:, xp.newaxis] * self._skill_sim_matrix)
            )
            / xp.sum(skill_weight_matrix)
        )
