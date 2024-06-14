from multiprocessing import Pool, Process
from typing import Any

import numpy as np
from tqdm import tqdm

from SkillSim import SkillSim
from SkillGroup import SkillGroup
from utils.MatrixSubsetIndexes import MatrixSubsetIndexes
from utils.TwoDimDict import TwoDimDict


class SkillSimCalculatorV2(SkillSim):
    skill_group: SkillGroup

    def __init__(self, skill_group: SkillGroup):
        self.skill_group = skill_group

    def rca(self, skill_index: int, job_index: int) -> float:
        # RCA numerator
        is_skill_in_job = self.skill_group.matrix[job_index][skill_index]

        if is_skill_in_job == 0:
            return 0

        num_skills_in_job = np.sum(self.skill_group.matrix[job_index])

        # RCA denominator
        num_jobs_with_skill = np.sum(self.skill_group.matrix[:, skill_index])
        num_skills = np.sum(self.skill_group.matrix)
        
        # print("RCA:", is_skill_in_job, num_skills_in_job, num_skills_in_job, num_skills)

        return (is_skill_in_job / num_skills_in_job) / (
            num_jobs_with_skill / num_skills
        )

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

    def cooccurrence_matrix(self) -> np.ndarray[Any, np.dtype[np.int8]]:
        skill_pair_indexes = []

        num_skills = self.skill_group.matrix.shape[1]

        for row_index in range(num_skills):
            for col_index in range(num_skills - row_index - 1):
                skill_pair_indexes.append([row_index, col_index + row_index + 1])

        with Pool() as pool:
            results = pool.imap(
                self.rca_cooccurrence_by_skill_pair,
                skill_pair_indexes,
            )

            cooccurrence_matrix = np.zeros((num_skills, num_skills))

            for i, cooccurrence_val in enumerate(results):
                [index1, index2] = skill_pair_indexes[i]
                cooccurrence_matrix[index1][index2] = cooccurrence_val
                cooccurrence_matrix[index2][index1] = cooccurrence_val

            return cooccurrence_matrix

    def skill_weight(
        self, matrix_subset: MatrixSubsetIndexes, skill_index: int
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

    def skill_set_similiarity(
        self,
        matrix_subset1: MatrixSubsetIndexes,
        matrix_subset2: MatrixSubsetIndexes,
    ) -> float:
        cooccurence_matrix = []
        skill1_weights = []
        skill2_weights = []

        skills_to_cooccurrence_val = TwoDimDict()

        pbar = tqdm(
            total=len(matrix_subset1) * len(matrix_subset2),
            desc="Creating Coccur Matrix",
        )

        for skill1_index in matrix_subset1:
            skill1_weights.append(self.skill_weight(matrix_subset1, skill1_index))

            cooccurence_vector = []

            for skill2_index in matrix_subset2:
                if len(skill2_weights) != len(matrix_subset2):
                    skill2_weights.append(
                        np.array([self.skill_weight(matrix_subset2, skill2_index)])
                    )

                cooccurrence_val_memo = skills_to_cooccurrence_val.get_value(
                    skill1_index, skill2_index
                )

                if cooccurrence_val_memo is None:
                    cooccurrence_val = self.rca_cooccurrence(skill1_index, skill2_index)

                    cooccurence_vector.append(cooccurrence_val)
                    skills_to_cooccurrence_val.add_key_val_pair(
                        skill1_index, skill2_index, cooccurrence_val
                    )
                else:
                    cooccurence_vector.append(cooccurrence_val_memo)

                pbar.update(1)

            cooccurence_matrix.append(np.array(cooccurence_vector))

        pbar.close()

        return np.matmul(
            np.array(skill1_weights),
            np.matmul(np.array(skill2_weights), np.array(cooccurence_matrix)),
        )
