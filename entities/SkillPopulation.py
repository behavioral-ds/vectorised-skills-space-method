import os
import json
import pickle
import shutil
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from utils.MatrixSubsetIndexes import MatrixSubsetIndexes
from utils.gzip import gzip_directory, uncompress_gzip

from entities.SkillGroup import SkillGroup
from entities.SkillSet import SkillSet
from entities.SkillGroupMetadata import SkillGroupMetadata
from entities.SkillSetMetadata import SkillSetMetadata


class SkillPopulation:
    matrix: NDArray[np.int8]  # matrix is a 2D array of only 1s and 0s
    skill_group_subsets: list[
        tuple[SkillGroupMetadata, MatrixSubsetIndexes]
    ]  # skill is 1-M with rows in matrix
    skill_sets_metadata: list[SkillSetMetadata | None]
    skill_names: list[str]  # 1-1 for each matrix column
    removed_skills: set[
        str
    ]  # skills removed from original input (did not reach threshold frequency)

    def __init__(
        self,
        file_path: str | None = None,
        skill_group_skills: dict[str, SkillGroup] | None = None,
        skill_freq_threshold: int | None = 5,
    ):
        if file_path is None and skill_group_skills is None:
            raise Exception(
                "Either a file path to a saved population or skill group skills dictionary needs to be provided."
            )

        if file_path is not None:
            uncompress_gzip(file_path)

            file_path = file_path.replace(".tar.gz", "")

            self.matrix = np.load(f"{file_path}/matrix.npy")

            with open(f"{file_path}/skill_names.json", "r") as f:
                self.skill_names = json.load(f)

            with open(f"{file_path}/removed_skills.json", "r") as f:
                self.removed_skills = set(json.load(f))

            with open(f"{file_path}/skill_group_subsets.pkl", "rb") as f:
                self.skill_group_subsets = pickle.load(f)

            with open(f"{file_path}/skill_sets_metadata.pkl", "rb") as f:
                self.skill_sets_metadata = pickle.load(f)

            shutil.rmtree(file_path)

            return

        if skill_group_skills is None:
            return

        skill_names, removed_skills = self.__filter_skills_by_threshold(
            skill_group_skills, skill_freq_threshold=5
        )
        self.skill_names = skill_names
        self.removed_skills = removed_skills

        num_unique_skills = len(self.skill_names)

        job_skill_matrix: list[list[int]] = []
        self.skill_group_subsets = []
        self.skill_sets_metadata = []
        row_index = 0

        for skill_group_id, skill_group_data in tqdm(
            skill_group_skills.items(), desc="Creating Skill Population"
        ):
            skill_group, skill_sets = self.__get_skill_group_with_sets(
                skill_group_id, skill_group_data
            )
            skill_vectors = []

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

            if len(skill_vectors) != 0:
                end_subset_index = row_index + len(skill_vectors) - 1

                self.skill_group_subsets.append(
                    (skill_group, MatrixSubsetIndexes((row_index, end_subset_index)))
                )

                row_index = end_subset_index + 1
                job_skill_matrix += skill_vectors

        self.matrix = np.array(job_skill_matrix, dtype=np.int8)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        object.__setattr__(
            self, "matrix", None
        )  # bypass type checking which self.matrix = None does not
        object.__setattr__(self, "skill_group_subsets", None)
        object.__setattr__(self, "skill_sets_metadata", None)
        object.__setattr__(self, "skill_names", None)
        object.__setattr__(self, "removed_skills", None)

    def save(self, file_path: str, output_name: str):
        population_path = f"{file_path}/{output_name}"
        os.makedirs(population_path, exist_ok=True)

        np.save(f"{population_path}/matrix.npy", self.matrix)

        with open(f"{population_path}/skill_names.json", "w") as f:
            json.dump(self.skill_names, f, indent=2)

        with open(f"{population_path}/removed_skills.json", "w") as f:
            json.dump(list(self.removed_skills), f, indent=2)

        with open(f"{population_path}/skill_group_subsets.pkl", "wb") as f:
            pickle.dump(self.skill_group_subsets, f)

        with open(f"{population_path}/skill_sets_metadata.pkl", "wb") as f:
            pickle.dump(self.skill_sets_metadata, f)

        gzip_directory(population_path, f"{file_path}/{output_name}.tar.gz")

        shutil.rmtree(population_path)

    def get_matrix_subset_by_sg(
        self, filter_func: Callable[[SkillGroupMetadata], bool]
    ) -> MatrixSubsetIndexes:
        matrix_subset = MatrixSubsetIndexes()

        for skill_group, group_matrix_subset in self.skill_group_subsets:
            if filter_func(skill_group):
                if matrix_subset is None:
                    matrix_subset = group_matrix_subset
                else:
                    matrix_subset += group_matrix_subset

        return matrix_subset

    def get_matrix_subset_by_ss_meta(
        self, filter_func: Callable[[SkillSetMetadata], bool]
    ) -> MatrixSubsetIndexes:
        indexes = []

        for i, skill_set_metadata in enumerate(self.skill_sets_metadata):
            if skill_set_metadata is not None and filter_func(skill_set_metadata):
                indexes.append(i)

        return MatrixSubsetIndexes(indexes)

    def __filter_skills_by_threshold(
        self,
        skill_group_skills: dict[str, SkillGroup],
        skill_freq_threshold: int,
    ) -> tuple[list[str], set[str]]:
        # counting number of occurrences for each skill
        skill_frequency: dict[str, int] = {}

        for skill_sets_data in tqdm(
            skill_group_skills.values(), desc="Removing Skills below Threshold"
        ):
            skill_sets = skill_sets_data["skill_sets"]

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
        self, skill_group_id: str, skill_group: SkillGroup
    ) -> tuple[SkillGroupMetadata, list[SkillSet]]:
        skill_group_properties = (
            {} if "properties" not in skill_group else skill_group["properties"]
        )

        skill_group_metadata = SkillGroupMetadata(
            skill_group_id,
            {
                "name": skill_group["name"],
            }
            | skill_group_properties,
        )

        return (
            skill_group_metadata,
            skill_group["skill_sets"],
        )

    def __get_skill_set_with_metadata(
        self, skill_set: SkillSet
    ) -> tuple[SkillSetMetadata | None, list[str]]:
        id = skill_set["id"]
        id_source = skill_set["id_source"] if "id_source" in skill_set else None
        properties = skill_set["properties"] if "properties" in skill_set else None

        return SkillSetMetadata(id, id_source, properties), skill_set["skills"]
