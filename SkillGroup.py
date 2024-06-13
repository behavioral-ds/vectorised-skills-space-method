from typing import Any

import numpy as np


class SkillGroup:
    matrix: np.ndarray[Any, np.dtype[np.int8]]
    skill_group_names: list[str]
    skill_names: list[str]
    removed_skills: set[str]

    def __init__(
        self,
        skill_group_skills: dict[str, list[list[str]]],
        skill_freq_threshold: int = 5,
    ):
        skill_frequency: dict[str, int] = {}

        for skill_sets in skill_group_skills.values():
            for skill_set in skill_sets:
                for skill in skill_set:
                    skill_frequency[skill] = skill_frequency.get(skill, 0) + 1

        removed_skills = set(
            [s for s in skill_frequency if skill_frequency[s] < skill_freq_threshold]
        )

        for skill in removed_skills:
            del skill_frequency[skill]

        population_skill_set = list(skill_frequency.keys())

        num_unique_skills = len(population_skill_set)

        job_skill_matrix = []
        self.skill_group_names = []
        self.skill_names = []

        for skill_group_name, skill_sets in skill_group_skills.items():
            for skill_set in skill_sets:
                has_skill = False
                skill_vector = np.zeros(num_unique_skills)

                for skill in skill_set:
                    if skill not in removed_skills:
                        has_skill = True
                        skill_vector[population_skill_set.index(skill)] = 1
                        self.skill_names.append(skill)

                if has_skill:
                    job_skill_matrix.append(skill_vector)
                    self.skill_group_names.append(skill_group_name)

        self.matrix = np.array(job_skill_matrix, dtype=np.int8)
        self.removed_skills = removed_skills
