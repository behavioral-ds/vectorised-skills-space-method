import cupy as cp

from SkillSim import SkillSim
from SkillGroup import SkillGroup
from utils.MatrixSubsetIndexes import MatrixSubsetIndexes


class SkillSimCalculatorV3(SkillSim):
    _skill_group: SkillGroup
    _skill_group_matrix: cp.ndarray

    _rca_matrix: cp.ndarray | None
    _skill_sim_matrix: cp.ndarray | None

    def __init__(self, skill_group: SkillGroup):
        self._skill_group = skill_group
        self._skill_group_matrix = cp.asarray(skill_group.matrix)
        self._rca_matrix = None
        self._skill_sim_matrix = None

    def calc_rca_matrix(self):
        num_skills_in_jobs = cp.sum(self._skill_group_matrix, axis=1)[:, cp.newaxis]

        nums_jobs_with_skill = cp.sum(self._skill_group_matrix, axis=0)
        num_skills = cp.sum(self._skill_group_matrix)

        self._rca_matrix = (self._skill_group_matrix / num_skills_in_jobs) / (
            nums_jobs_with_skill / num_skills
        )

    def calc_skill_sim_matrix(self):
        effective_use_matrix = cp.where(self._rca_matrix >= 1.0, 1.0, 0.0)

        effective_use_sum_matrix = cp.matmul(
            effective_use_matrix.T, effective_use_matrix
        )

        skill_count_vector = cp.diag(effective_use_sum_matrix)

        t1 = cp.tile(skill_count_vector, (len(skill_count_vector), 1))
        t2 = t1.T

        effective_use_sum_max = cp.maximum(t1, t2)

        self._skill_sim_matrix = effective_use_sum_matrix / effective_use_sum_max

    def get_skill_sim_matrix(self) -> cp.ndarray | None:
        return self._skill_sim_matrix

    def skill_set_one_hot_vector(
        self, matrix_subset: MatrixSubsetIndexes
    ) -> cp.ndarray:
        # rename of previous calcv2 skill_set_vector method
        skill_vector = cp.clip(
            cp.sum(self._skill_group_matrix[matrix_subset.indexes], axis=0), None, 1
        )

        return skill_vector

    def skill_weight_vector(self, matrix_subset: MatrixSubsetIndexes) -> cp.ndarray:
        return cp.sum(
            self.skill_set_one_hot_vector(matrix_subset)
            * self._rca_matrix[matrix_subset.indexes],
            axis=0,
        ) / len(matrix_subset)

    def skill_set_similarity(
        self, matrix_subset_1: MatrixSubsetIndexes, matrix_subset_2: MatrixSubsetIndexes
    ) -> float:
        if self._rca_matrix is None:
            self.calc_rca_matrix()

        if self._skill_sim_matrix is None:
            self.calc_skill_sim_matrix()

        # skills that aren't included in the skill set of subset 1 or 2 will have a weight of zero
        # when the element-wise dot product is calculated with the skill sim matrix this means
        # skills that weren't included in the subset will be zeroed out and not count towards the sum
        subset_1_weight_vector = self.skill_weight_vector(matrix_subset_1)
        subset_2_weight_vector = self.skill_weight_vector(matrix_subset_2)

        _, num_skills = self._skill_sim_matrix.shape
        skill_weight_matrix = cp.ones((num_skills, num_skills))
        skill_weight_matrix = (
            subset_1_weight_vector[:, cp.newaxis] * skill_weight_matrix
        )
        skill_weight_matrix = subset_2_weight_vector * skill_weight_matrix

        return cp.sum(
            subset_2_weight_vector
            * (subset_1_weight_vector[:, cp.newaxis] * self._skill_sim_matrix)
        ) / cp.sum(skill_weight_matrix)
