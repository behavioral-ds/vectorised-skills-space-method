from typing import Any, Optional, Callable, Type
from dataclasses import dataclass

import numpy as np

from utils.MatrixSubsetIndexes import MatrixSubsetIndexes


@dataclass
class SkillSetMetadata:
    id: Optional[str] = None
    id_source: Optional[str] = None
    properties: Optional[dict] = None


@dataclass
class SkillGroup:
    id: str
    properties: Optional[dict] = None


class SkillPopulation:
    matrix: np.ndarray[Any, np.dtype[np.int8]]  # matrix is a 2D array of only 1s and 0s
    skill_group_subsets: list[
        tuple[SkillGroup, MatrixSubsetIndexes]
    ]  # skill is 1-M with rows in matrix
    skill_sets_metadata: list[SkillSetMetadata | None]
    skill_names: list[str]  # 1-1 for each matrix column
    removed_skills: set[
        str
    ]  # skills removed from original input (did not reach threshold frequency)

    def __init__(
        self,
        skill_group_skills: dict[str, list[list[str]]],
        skill_freq_threshold: int = 5,
    ):
        skill_names, removed_skills = self.__filter_skills_by_threshold(
            skill_group_skills, skill_freq_threshold
        )
        self.skill_names = skill_names
        self.removed_skills = removed_skills

        num_unique_skills = len(self.skill_names)

        job_skill_matrix = []
        self.skill_group_subsets = []
        self.skill_sets_metadata = []
        row_index = 0

        for skill_group_id, skill_group_data in skill_group_skills.items():
            skill_group, skill_sets = self.__get_skill_group_with_sets(
                skill_group_id, skill_group_data
            )
            skill_vectors = []
            num_skill_sets_removed = 0

            for skill_set_data in skill_sets:
                skill_set_metadata, skills = self.__get_skill_set_with_metadata(
                    skill_set_data
                )
                has_skill = False
                skill_vector = np.zeros(num_unique_skills)

                for skill in skills:
                    if skill not in self.removed_skills:
                        has_skill = True
                        skill_vector[self.skill_names.index(skill)] = 1

                if has_skill:
                    skill_vectors.append(skill_vector)
                    self.skill_sets_metadata.append(skill_set_metadata)
                else:
                    num_skill_sets_removed += 1

            if len(skill_vectors) != 0:
                end_subset_index = row_index + len(skill_sets) - num_skill_sets_removed
                self.skill_group_subsets.append(
                    (skill_group, MatrixSubsetIndexes((row_index, end_subset_index)))
                )
                row_index = end_subset_index + 1
                job_skill_matrix += skill_vectors

        self.matrix = np.array(job_skill_matrix, dtype=np.int8)

    def __filter_skills_by_threshold(
        self,
        skill_group_skills: dict[str, list[list[str]]],
        skill_freq_threshold: int = 5,
    ) -> tuple[list[str], list[str]]:
        # counting number of occurrences for each skill
        skill_frequency: dict[str, int] = {}

        for skill_sets_data in skill_group_skills.values():
            skill_sets = (
                skill_sets_data
                if type(skill_sets_data) is list
                else skill_sets_data["skill_sets"]
            )

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

        return list(skill_frequency.keys()), removed_skills

    def __get_skill_group_with_sets(
        self, skill_group_id: str, skill_group_data: list | dict
    ) -> tuple[SkillGroup, list[list[str]]]:
        if type(skill_group_data) is list:
            return (SkillGroup(skill_group_id), skill_group_data)

        return (
            SkillGroup(
                skill_group_id,
                {
                    key: value
                    for key, value in skill_group_data.items()
                    if key != "skill_sets"
                },
            ),
            skill_group_data["skill_sets"],
        )

    def __get_skill_set_with_metadata(
        self, skill_set_data: list | dict
    ) -> tuple[SkillSetMetadata | None, list[str]]:
        if type(skill_set_data) is list:
            return (None, skill_set_data)

        id = skill_set_data["id"] if "id" in skill_set_data else None
        id_source = (
            skill_set_data["id_source"] if "id_source" in skill_set_data else None
        )
        properties = (
            skill_set_data["properties"] if "properties" in skill_set_data else None
        )

        return SkillSetMetadata(id, id_source, properties), skill_set_data["skills"]

    def get_matrix_subset_by_sg(
        self, filter_func: Callable[[Type[SkillGroup]], bool]
    ) -> MatrixSubsetIndexes:
        matrix_subset = None

        for skill_group, group_matrix_subset in self.skill_group_subsets:
            if filter_func(skill_group):
                if matrix_subset is None:
                    matrix_subset = group_matrix_subset
                else:
                    matrix_subset += group_matrix_subset

        return matrix_subset

    def get_matrix_subset_by_ss_meta(
        self, filter_func: Callable[[Type[SkillSetMetadata]], bool]
    ) -> MatrixSubsetIndexes:
        indexes = []

        for i, skill_set_metadata in enumerate(self.skill_sets_metadata):
            if filter_func(skill_set_metadata):
                indexes.append(i)

        return MatrixSubsetIndexes(indexes)
