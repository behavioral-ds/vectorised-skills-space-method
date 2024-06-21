from multiprocessing import Pool, Process
from typing import Any

import cupy as cp
from tqdm import tqdm

from SkillSim import SkillSim
from SkillGroup import SkillGroup
from utils.MatrixSubsetIndexes import MatrixSubsetIndexes
from utils.TwoDimDict import TwoDimDict


class SkillSimCalculatorV3(SkillSim):
    _skill_group: SkillGroup
    _skill_group_matrix: cp.ndarray

    _rca_matrix: cp.ndarray | None
    _skill_sim_matrix: cp.ndarray | None

    def __init__(self, skill_group: SkillGroup):
        self._skill_group = skill_group
        self._skill_group_matrix = cp.asarray(skill_group.matrix)
        self._skill_sim_matrix = None

    def calc_rca_matrix(self):
        num_skills_in_jobs = cp.sum(self._skill_group_matrix, axis=1)
        nums_jobs_with_skill = 1 / cp.sum(self._skill_group_matrix, axis=0)
        num_skills = cp.sum(self._skill_group_matrix)

        self._rca_matrix = (
            self._skill_group_matrix * num_skills_in_jobs * nums_jobs_with_skill
        ) / num_skills

    def calc_skill_sim_matrix(self):
        effective_use_matrix = cp.where(self._rca_matrix >= 1, 1, 0)

        effective_use_sum_matrix = cp.matmul(
            effective_use_matrix.T, effective_use_matrix
        )

        skill_count_vector = cp.diag(effective_use_sum_matrix)

        t1 = cp.tile(skill_count_vector, (len(skill_count_vector), 1))
        t2 = t1.T

        effective_use_sum_max = cp.maximum(t1, t2)

        self._skill_sim_matrix = effective_use_sum_matrix / effective_use_sum_max

    def get_skill_weight(self, matrix_subset: MatrixSubsetIndexes) -> cp.ndarray:
        return cp.sum(self._rca_matrix[matrix_subset]) / len(matrix_subset)

    def get_skill_set_similarity(
        self, matrix_subset1: MatrixSubsetIndexes, matrix_subset2: MatrixSubsetIndexes
    ) -> float:
        pass
