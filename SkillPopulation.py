from typing import Any, Optional
from dataclasses import dataclass

import numpy as np

from utils.MatrixSubsetIndexes import MatrixSubsetIndexes


@dataclass
class SkillSet:
    id: Optional[str] = None
    id_source: Optional[str] = None
    properties: Optional[dict] = None


@dataclass
class SkillGroup:
    id: str
    properties: Optional[dict] = None


class SkillPopulation:
    matrix: np.ndarray[Any, np.dtype[np.int8]]  # matrix is a 2D array of only 1s and 0s
    skill_groups_subset: list[tuple[SkillGroup, MatrixSubsetIndexes]]
    skill_names: list[str]
    removed_skills: set[str]

    def __init__(
        self,
        skill_group_skills: dict[str, list[list[str]]],
        skill_freq_threshold: int = 5,
    ):
        # counting number of occurrences for each skill
        skill_frequency: dict[str, int] = {}

        for skill_sets in skill_group_skills.values():
            for skill_set in skill_sets:
                skills = skill_set if type(skill_set) is list else skill_set["skills"]

                for skill in skills:
                    skill_frequency[skill] = skill_frequency.get(skill, 0) + 1

        # removing skills that have a frequency less than the threshold frequency
        removed_skills = set(
            [s for s in skill_frequency if skill_frequency[s] < skill_freq_threshold]
        )

        for skill in removed_skills:
            del skill_frequency[skill]

        self.skill_names = list(skill_frequency.keys())

        num_unique_skills = len(self.skill_names)

        job_skill_matrix = []
        self.skill_groups_subset = []
        row_index = 0

        for skill_group_id, skill_sets in skill_group_skills.items():
            skill_group = SkillGroup(skill_group_id)
            num_skill_sets_removed = 0
            skill_vectors = []

            for skill_set in skill_sets:
                skills = skill_set if type(skill_set) is list else skill_set["skills"]

                has_skill = False
                skill_vector = np.zeros(num_unique_skills)

                for skill in skills:
                    if skill not in removed_skills:
                        has_skill = True
                        skill_vector[self.skill_names.index(skill)] = 1

                if has_skill:
                    skill_vectors.append(skill_vector)
                else:
                    num_skill_sets_removed += 1

            if len(skill_vectors) != 0:
                end_subset_index = row_index + len(skill_sets) - num_skill_sets_removed
                self.skill_groups_subset.append(
                    skill_group, MatrixSubsetIndexes(row_index, end_subset_index)
                )
                row_index = end_subset_index + 1
                job_skill_matrix += skill_vectors

        self.matrix = np.array(job_skill_matrix, dtype=np.int8)
        self.removed_skills = removed_skills
