from multiprocessing import Pool, Process
from typing import Any

import numpy as np
from tqdm import tqdm

from sim_calculators.SkillSim import SkillSim
from SkillGroup import SkillGroup
from utils.MatrixSubsetIndexes import MatrixSubsetIndexes
from utils.TwoDimDict import TwoDimDict


class SkillSimCalculatorV2(SkillSim):
    skill_group: SkillGroup
    cooccurrence_matrix: np.ndarray[Any, np.dtype[np.int8]] | None
    num_skills_in_job: dict[int, int]
    num_jobs_with_skill: dict[int, int]
    num_skills: int
    rca_memo: dict[tuple[int, int], float]

    def __init__(self, skill_group: SkillGroup):
        self.skill_group = skill_group
        self.cooccurrence_matrix = None
        self.num_skills_in_job = {}
        self.num_jobs_with_skill = {}
        self.num_skills = np.sum(skill_group.matrix)
        self.rca_memo = {}

    def rca(self, skill_index: int, job_index: int) -> float:
        if (skill_index, job_index) in self.rca_memo:
            return self.rca_memo[(skill_index, job_index)]

        # RCA numerator
        is_skill_in_job = self.skill_group.matrix[job_index][skill_index]

        if is_skill_in_job == 0:
            return 0

        num_skills_in_job = None

        if job_index in self.num_skills_in_job:
            num_skills_in_job = self.num_skills_in_job[job_index]
        else:
            num_skills_in_job = np.sum(self.skill_group.matrix[job_index])
            self.num_skills_in_job[job_index] = num_skills_in_job

        # RCA denominator
        num_jobs_with_skill = None

        if skill_index in self.num_jobs_with_skill:
            num_jobs_with_skill = self.num_jobs_with_skill[skill_index]
        else:
            num_jobs_with_skill = np.sum(self.skill_group.matrix[:, skill_index])
            self.num_jobs_with_skill[skill_index] = num_jobs_with_skill

        num_skills = self.num_skills

        rca_val = (is_skill_in_job / num_skills_in_job) / (
            num_jobs_with_skill / num_skills
        )

        self.rca_memo[(skill_index, job_index)] = rca_val

        return rca_val

    def rca_cooccurrence(self, skill1_index: int, skill2_index: int) -> float:
        """
        Calculates the likelihood of two skills are "effectively" (i.e. have an advantage) in the same
        skill groups (e.g. a job).

        Args:
            skill1_index (int): the column/element index for a given skill (in the nxn matrix).
            skill2_index (int): the column/element index for a given skill (in the nxn matrix).

        Returns:
            float: probability that skill1 and skill2 occur in the same job ads.
        """
        skill_pairwise_total = 0
        skill1_total_effective_use = 0
        skill2_total_effective_use = 0

        for job_index in range(self.skill_group.matrix.shape[0]):
            is_skill1_effective = 1 if self.rca(skill1_index, job_index) >= 1 else 0
            is_skill2_effective = 1 if self.rca(skill2_index, job_index) >= 1 else 0

            skill_pairwise_total += is_skill1_effective * is_skill2_effective
            skill1_total_effective_use += is_skill1_effective
            skill2_total_effective_use += is_skill2_effective

        if skill1_total_effective_use == 0 and skill2_total_effective_use == 0:
            return 0

        return skill_pairwise_total / (
            skill1_total_effective_use
            if skill1_total_effective_use > skill2_total_effective_use
            else skill2_total_effective_use
        )

    def rca_cooccurrence_by_skill_pair(self, skill_pair_index):
        return self.rca_cooccurrence(skill_pair_index[0], skill_pair_index[1])

    def calc_cooccurrence_matrix(self) -> np.ndarray[Any, np.dtype[np.int8]]:
        skill_pair_indexes = []

        _, num_skills = self.skill_group.matrix.shape

        for row_index in range(num_skills):
            for col_index in range(num_skills - row_index):
                skill_pair_indexes.append([row_index, col_index + row_index])

        with Pool() as pool:
            results = tqdm(
                pool.imap(
                    self.rca_cooccurrence_by_skill_pair,
                    skill_pair_indexes,
                ),
                total=len(skill_pair_indexes),
                desc="Co-occur Matrix",
            )

            cooccurrence_matrix = np.zeros((num_skills, num_skills))

            for i, cooccurrence_val in enumerate(results):
                [index1, index2] = skill_pair_indexes[i]
                cooccurrence_matrix[index1][index2] = cooccurrence_val
                cooccurrence_matrix[index2][index1] = cooccurrence_val

            self.cooccurrence_matrix = cooccurrence_matrix

            return cooccurrence_matrix

    def set_cooccurrence_matrix(
        self, cooccurrence_matrix: np.ndarray[Any, np.dtype[np.int8]]
    ):
        self.cooccurrence_matrix = cooccurrence_matrix

    def skill_weight(
        self, skill_index: int, matrix_subset: MatrixSubsetIndexes
    ) -> float:
        """
        Calculates the weight (importance) of a skill for a given skill set.

        Args:
            matrix_subset (np.ndarray[Any, np.dtype[np.int8]]): the vectors of a given skill group.
            skill_index (int): the column/element index for a given skill (in the 2x2 matrix).

        Returns:
            float: the weight/importance of a given skill in a skill set.
        """

        return np.sum(
            np.array([self.rca(skill_index, job_index) for job_index in matrix_subset])
        ) / len(matrix_subset)

    def skill_weight_by_skill_subset_pair(
        self, skill_subset_pair: tuple[int, MatrixSubsetIndexes]
    ) -> float:
        skill_index, matrix_subset = skill_subset_pair
        return self.skill_weight(skill_index, matrix_subset)

    def skill_weights(
        self, matrix_subset: MatrixSubsetIndexes
    ) -> np.ndarray[Any, np.dtype[np.int8]]:
        _, num_skills = self.skill_group.matrix.shape

        index_subset_pairs = [
            (skill_index, matrix_subset) for skill_index in range(num_skills)
        ]

        with Pool() as pool:
            results = tqdm(
                pool.imap(self.skill_weight_by_skill_subset_pair, index_subset_pairs),
                total=len(index_subset_pairs),
                desc="Skill Weights",
            )
            return np.array(list(results))

    def skill_set_vector(
        self, matrix_subset1: MatrixSubsetIndexes, matrix_subset2: MatrixSubsetIndexes
    ) -> np.ndarray[Any, np.dtype[np.int8]]:
        skill_vector1 = np.clip(
            np.sum(self.skill_group.matrix[matrix_subset1.indexes], axis=0), None, 1
        )
        skill_vector2 = np.clip(
            np.sum(self.skill_group.matrix[matrix_subset2.indexes], axis=0), None, 1
        )

        return skill_vector1, skill_vector2

    def skill_set_similarity(
        self,
        matrix_subset_1: MatrixSubsetIndexes,
        matrix_subset_2: MatrixSubsetIndexes,
    ) -> float:
        skill_vector1, skill_vector2 = self.skill_set_vector(
            matrix_subset_1, matrix_subset_2
        )
        skill1_weight_matrix = skill_vector1 * self.skill_weights(matrix_subset_1)
        skill2_weight_matrix = skill_vector2 * self.skill_weights(matrix_subset_2)

        _, num_skills = self.skill_group.matrix.shape
        skill_weight_matrix = np.ones((num_skills, num_skills))
        skill_weight_matrix = skill1_weight_matrix[:, np.newaxis] * skill_weight_matrix
        skill_weight_matrix = skill2_weight_matrix * skill_weight_matrix

        skill_cooccurrence_matrix = (
            self.calc_cooccurrence_matrix()
            if self.cooccurrence_matrix is None
            else self.cooccurrence_matrix
        )

        return np.sum(skill_weight_matrix * skill_cooccurrence_matrix) / np.sum(
            skill_weight_matrix
        )
