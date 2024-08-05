import unittest
from typing import Any

import numpy as np

from utils import get_random_skill_group, get_skill_group_matrix
from sim_calculators.SkillSimCalculatorBaseline import get_job_population


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
