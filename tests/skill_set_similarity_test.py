import unittest
import random
import time

import numpy as np

from tests.random_data.get_random_skills import (
    get_random_skill_group,
    get_random_occ_to_skills,
)
from utils import MatrixSubsetIndexes, get_job_population, xp
from entities import SkillPopulationDeprecated, SkillPopulation
from sim_calculators import (
    SkillSimCalculatorBaseline,
    SkillSimCalculatorV2,
    SkillSimCalculatorV3,
)


class TestSkillSimCalculators(unittest.TestCase):
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

    def test_sss_v2_with_baseline(self):
        num_occupations = 15
        num_skills = 5

        print("Skill Set Similarity Test (Baseline & V2)")

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
            baseline_result = skill_sim_calc.skill_set_similarity(
                job_population[start_index:end_index],
                [job_population[i] for i in matrix_subset_2.indexes],
            )
            end = time.time()
            baseline_time = round((end - start) * 1000, 2)

            start = time.time()
            v2_result = skill_sim_calc_2.skill_set_similarity(
                matrix_subset_1, matrix_subset_2
            )
            end = time.time()
            v2_time = round((end - start) * 1000, 2)

            print("Baseline Execution Time:", baseline_time, "milliseconds")
            print("V2 Execution Time:", v2_time, "milliseconds")

            with self.subTest():
                # TODO: Update baseline and v2 to use numpy floats so they produce exactly equal floats
                self.assertLessEqual(
                    abs(int(baseline_result * 100) - int(v2_result * 100)), 1
                )

    def test_skill_sim_matrix_v3(self):
        num_occupations = 15
        num_skills = 5

        print("Skill Set Similarity Test (V2 & V3)")

        for test_num in range(1, 6):
            skill_group = get_random_skill_group(
                num_occupations * test_num, num_skills * test_num
            )
            job_population = get_job_population(skill_group)

            print(
                f"Test {test_num}: Occupations = {num_occupations * test_num}, Jobs = {len(job_population)}, Skills = {num_skills * test_num}"
            )

            skill_sim_calc_v2 = SkillSimCalculatorV2(skill_group)
            skill_sim_calc_v3 = SkillSimCalculatorV3(skill_group)

            start = time.time()
            v2_result = skill_sim_calc_v2.calc_cooccurrence_matrix()
            end = time.time()
            v2_time = round((end - start) * 1000, 2)

            start = time.time()
            skill_sim_calc_v3.calc_rca_matrix()
            skill_sim_calc_v3.calc_skill_sim_matrix()
            v3_result = skill_sim_calc_v3.get_skill_sim_matrix()
            end = time.time()
            v3_time = round((end - start) * 1000, 2)

            print("V2 Execution Time:", v2_time, "milliseconds")
            print("V3 Execution Time:", v3_time, "milliseconds")

            with self.subTest():
                self.assertTrue(xp.array_equal(v3_result, xp.asarray(v2_result)))

    def test_sss_v3_with_v2(self):
        num_occupations = 90
        num_skills = 40

        print("Skill Set Similarity Test (V2 & V3)")

        for test_num in range(1, 6):
            occ_to_skills = get_random_occ_to_skills(
                num_occupations * test_num, num_skills * test_num
            )

            skill_group = SkillPopulationDeprecated(occ_to_skills)
            skill_population = SkillPopulation(skill_group_skills=occ_to_skills)

            job_population = get_job_population(skill_group)

            print(
                f"Test {test_num}: Occupations = {num_occupations * test_num}, Jobs = {len(job_population)}, Skills = {num_skills * test_num}"
            )

            skill_sim_calc_v2 = SkillSimCalculatorV2(skill_group)
            skill_sim_calc_v3 = SkillSimCalculatorV3(skill_population)

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
            v2_result = skill_sim_calc_v2.skill_set_similarity(
                matrix_subset_1, matrix_subset_2
            )
            end = time.time()
            v2_time = round((end - start) * 1000, 2)

            start = time.time()
            v3_result = skill_sim_calc_v3.skill_set_similarity(
                matrix_subset_1, matrix_subset_2
            )
            end = time.time()
            v3_time = round((end - start) * 1000, 2)

            print("V2 Execution Time:", v2_time, "milliseconds")
            print("V3 Execution Time:", v3_time, "milliseconds")

            with self.subTest():
                # TODO: Update v2 to use numpy floats so they produce exactly equal floats
                self.assertAlmostEqual(v2_result, v3_result, 6)

    def test_sss_v3_with_v3_sparse(self):
        num_occupations = 90
        num_skills = 40

        print("Skill Set Similarity Test (V3 & V3 Sparse)")

        for test_num in range(1, 6):
            occ_to_skills = get_random_occ_to_skills(
                num_occupations * test_num, num_skills * test_num
            )

            skill_group = SkillPopulationDeprecated(occ_to_skills)
            skill_population = SkillPopulation(skill_group_skills=occ_to_skills)

            job_population = get_job_population(skill_group)

            print(
                f"Test {test_num}: Occupations = {num_occupations * test_num}, Jobs = {len(job_population)}, Skills = {num_skills * test_num}"
            )

            skill_sim_calc_1 = SkillSimCalculatorV3(skill_population)
            skill_sim_calc_2 = SkillSimCalculatorV3(
                skill_population, use_sparse_matrices=True
            )

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
            v3_result = skill_sim_calc_1.skill_set_similarity(
                matrix_subset_1, matrix_subset_2
            )
            end = time.time()
            v3_time = round((end - start) * 1000, 2)

            start = time.time()
            v3_sparse_result = skill_sim_calc_2.skill_set_similarity(
                matrix_subset_1, matrix_subset_2
            )
            end = time.time()
            v3_sparse_time = round((end - start) * 1000, 2)

            print("V3 Execution Time:", v3_time, "milliseconds")
            print("V3 Sparse Execution Time:", v3_sparse_time, "milliseconds")

            with self.subTest():
                self.assertAlmostEqual(v3_result, v3_sparse_result, 6)


if __name__ == "__main__":
    unittest.main()
