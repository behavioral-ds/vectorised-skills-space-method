import unittest

import numpy as np

from utils.get_random_skill_group import get_random_skill_group
from SkillSimCalculatorBaseline import get_job_population, SkillSimCalculatorBaseline
from SkillSimCalculatorV2 import SkillSimCalculatorV2


class TestCooccurrenceMatrix(unittest.TestCase):
    def test_equal_matrices(self):
        is_equal = True

        for _ in range(5):
            skill_group = get_random_skill_group()
            skill_names = skill_group.skill_names
            job_population = get_job_population(skill_group)

            baseline_calc = SkillSimCalculatorBaseline(job_population, skill_names)
            calc_v2 = SkillSimCalculatorV2(skill_group)

            is_equal = np.array_equal(
                baseline_calc.cooccurrence_matrix(), calc_v2.cooccurrence_matrix()
            )

            if not is_equal:
                break

        self.assertTrue(is_equal)
