import unittest

import numpy as np

from entities import SkillGroup, SkillPopulation

from .random.get_random_skills import (
    get_random_occ_to_skills,
    get_random_skill_population,
)


def remove_duplicates(input_list):
    seen = set()
    result = []
    for item in input_list:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


class TestSkillPopulation(unittest.TestCase):
    def test_sp_equals_sg(self):
        for _ in range(5):
            occ_to_skills = get_random_occ_to_skills()

            skill_group = SkillGroup(occ_to_skills)
            skill_population_1 = get_random_skill_population(True, occ_to_skills)
            skill_population_2 = get_random_skill_population(True, occ_to_skills)

            with self.subTest():
                self.assertTrue(
                    np.array_equal(skill_group.matrix, skill_population_1.matrix)
                )

            with self.subTest():
                self.assertTrue(
                    np.array_equal(skill_group.matrix, skill_population_2.matrix)
                )

            with self.subTest():
                self.assertEqual(
                    skill_group.skill_names, skill_population_1.skill_names
                )

            with self.subTest():
                self.assertEqual(
                    skill_group.skill_names, skill_population_2.skill_names
                )

            with self.subTest():
                self.assertEqual(
                    skill_group.removed_skills, skill_population_1.removed_skills
                )

            with self.subTest():
                self.assertEqual(
                    skill_group.removed_skills, skill_population_2.removed_skills
                )

            with self.subTest():
                self.assertEqual(
                    remove_duplicates(skill_group.skill_group_names),
                    [x[0].id for x in skill_population_1.skill_group_subsets],
                )

            with self.subTest():
                self.assertEqual(
                    remove_duplicates(skill_group.skill_group_names),
                    [x[0].id for x in skill_population_2.skill_group_subsets],
                )

    def test_context_manager(self):
        occ_to_skills = get_random_occ_to_skills()

        skill_population_1 = SkillPopulation(occ_to_skills)

        with SkillPopulation(occ_to_skills) as skill_population_2:
            with self.subTest():
                self.assertTrue(
                    np.array_equal(skill_population_1.matrix, skill_population_2.matrix)
                )

            with self.subTest():
                self.assertEqual(
                    skill_population_1.skill_names, skill_population_2.skill_names
                )

            with self.subTest():
                self.assertEqual(
                    skill_population_1.removed_skills, skill_population_2.removed_skills
                )

            with self.subTest():
                is_equal = True

                for i, (skill_group_1, matrix_subset_1) in enumerate(
                    skill_population_1.skill_group_subsets
                ):
                    skill_group_2, matrix_subset_2 = (
                        skill_population_2.skill_group_subsets[i]
                    )

                    if skill_group_1.id != skill_group_2.id:
                        is_equal = False
                        break

                    if matrix_subset_1.indexes != matrix_subset_1.indexes:
                        is_equal = False
                        break

                self.assertTrue(is_equal)

        with self.subTest():
            self.assertIsNone(skill_population_2.matrix)
            self.assertIsNone(skill_population_2.skill_names)
            self.assertIsNone(skill_population_2.removed_skills)
            self.assertIsNone(skill_population_2.skill_group_subsets)


if __name__ == "__main__":
    unittest.main()
