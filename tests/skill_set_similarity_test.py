import unittest
import random
import time

import numpy as np

from utils.get_random_skill_group import get_random_skill_group
from utils.MatrixSubsetIndexes import MatrixSubsetIndexes
from SkillSimCalculatorBaseline import get_job_population, SkillSimCalculatorBaseline
from SkillSimCalculatorV2 import SkillSimCalculatorV2


class TestSkillSimCalculatorV2(unittest.TestCase):
    def test_set_cooccurrence_matrix(self):
        skill_group_1 = get_random_skill_group(15, 5)
        skill_group_2 = get_random_skill_group(15, 5)
        
        while skill_group_1 == skill_group_2:
            skill_group_2 = get_random_skill_group(15, 5)
        
        calc_1 = SkillSimCalculatorV2(skill_group_1)
        calc_1.calc_cooccurrence_matrix()
        
        calc_2 = SkillSimCalculatorV2(skill_group_2)
        cooccur_matrix_2 = calc_2.calc_cooccurrence_matrix()
        
        calc_1.set_cooccurrence_matrix(cooccur_matrix_2)
        
        self.assertTrue(np.array_equal(calc_1.cooccurrence_matrix, cooccur_matrix_2))
        

    def test_skill_set_similarity(self):
        num_occupations = 15
        num_skills = 5

        print("Skill Set Similarity Test")

        for test_num in range(1, 6):
            skill_group = get_random_skill_group(
                num_occupations * test_num, num_skills * test_num
            )
            job_population = get_job_population(skill_group)

            print(
                f"Test {test_num}: Occupations = {num_occupations * test_num}, Jobs = {len(job_population)}, Skills = {num_skills * test_num}"
            )

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

            start = time.time()
            baseline_result = skill_sim_calc.skill_set_similiarity(
                job_population[start_index:end_index],
                [job_population[i] for i in matrix_subset_2.indexes],
            )
            end = time.time()
            baseline_time = round((end - start) * 1000, 2)

            start = time.time()
            v2_result = skill_sim_calc_2.skill_set_similiarity(
                matrix_subset_1, matrix_subset_2
            )
            end = time.time()
            v2_time = round((end - start) * 1000, 2)

            print("Baseline Execution Time:", baseline_time, "milliseconds")
            print("V2 Execution Time:", v2_time, "milliseconds")

            with self.subTest():
                self.assertEqual(int(baseline_result * 100), int(v2_result * 100))
