import unittest
import random

import numpy as np

from utils.get_random_skill_group import get_random_skill_group
from utils.MatrixSubsetIndexes import MatrixSubsetIndexes
from SkillSimCalculatorBaseline import get_job_population, SkillSimCalculatorBaseline
from SkillSimCalculatorV2 import SkillSimCalculatorV2


class TestCooccurrenceMatrix(unittest.TestCase):
    def test_no_removed_skills(self):
        num_occupations = 15
        num_skills = 5

        for test_num in range(1, 6):
            skill_group = get_random_skill_group(
                num_occupations * test_num, num_skills * test_num
            )
            job_population = get_job_population(skill_group)

            skill_sim_calc = SkillSimCalculatorBaseline(
                job_population, skills=skill_group.skill_names
            )
            skill_sim_calc_2 = SkillSimCalculatorV2(skill_group)

            start_index = random.randint(0, len(job_population) - 2)
            end_index = random.randint(start_index + 1, len(job_population) - 1)
            matrix_subset_1 = MatrixSubsetIndexes((start_index, end_index))

            matrix_subset_2 = MatrixSubsetIndexes(
                list(
                    set(
                        [
                            random.randint(0, len(job_population) - 1)
                            for _ in range(random.randint(1, len(job_population)))
                        ]
                    )
                )
            )

            baseline_result = skill_sim_calc.skill_set_similiarity(
                job_population[start_index:end_index],
                [job_population[i] for i in matrix_subset_2.indexes],
            )
            v2_result = skill_sim_calc_2.skill_set_similiarity(
                matrix_subset_1, matrix_subset_2
            )

            with self.subTest():
                self.assertAlmostEqual(baseline_result, v2_result, 2)
