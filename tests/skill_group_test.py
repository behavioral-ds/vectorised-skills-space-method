import unittest
from typing import Any

import numpy as np

from utils.get_random_skill_group import get_random_skill_group
from SkillSimCalculatorBaseline import get_job_population, Job


def get_skill_group_matrix(
    job_population: list[Job], skill_names: list[str]
) -> np.ndarray[Any, np.dtype[np.int8]]:
    sg_matrix = np.zeros((len(job_population), len(skill_names)))

    for row_index, job in enumerate(job_population):
        for skill in job.skills:
            col_index = skill_names.index(skill.name)
            sg_matrix[row_index][col_index] = 1

    return sg_matrix


class TestSkillGroup(unittest.TestCase):
    def setUp(self):
        self.skill_group = get_random_skill_group()
        self.skill_names = self.skill_group.skill_names
        self.job_population = get_job_population(self.skill_group)

    def test_no_removed_skills(self):
        is_no_removed_skills = True

        for job in self.job_population:
            for skill in job.skills:
                if skill.name in self.skill_group.removed_skills:
                    is_no_removed_skills = False
                    break

        self.assertTrue(is_no_removed_skills)

    def test_equal_matrices(self):
        is_equal = True

        for _ in range(10):
            is_equal = np.array_equal(
                self.skill_group.matrix,
                get_skill_group_matrix(self.job_population, self.skill_names),
            )

            if not is_equal:
                break

        self.assertTrue(is_equal)
