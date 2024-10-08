import copy
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

from sim_calculators.SkillSim import SkillSim
from entities import SkillPopulationDeprecated
from entities import SkillPopulation
from utils import MatrixSubsetIndexes, xp


class SkillSimCalculatorV3(SkillSim):
    """Skill Similarity Calculator V3 implements the skill set similarity method using matrix operations
    from NumPy (CPU), CuPy (Nvidia Cuda) or MLX (Apple). Given these dependencies are conditional to runtime,
    the xp wrapper is used, which unforenately negates type hints. Moreover, Numpy's NDArray is used as a type
    hint for methods that return arrays. CuPy and MLX arrays are largely similar and compatible however issues
    could arise in terms of functionality and output.
    """

    _skill_population_matrix: NDArray[np.int8] | csr_array
    _rca_matrix: NDArray[np.float64] | csr_array
    _skill_sim_matrix: NDArray[np.float64]
    _use_sparse_matrices: bool

    def __init__(
        self,
        skill_population: SkillPopulation | SkillPopulationDeprecated,
        use_sparse_matrices: bool = False,
    ):
        self._skill_population_matrix = (
            csr_array(skill_population.matrix)
            if use_sparse_matrices
            else xp.asarray(skill_population.matrix)
        )
        self._rca_matrix = None
        self._skill_sim_matrix = None
        self._use_sparse_matrices = use_sparse_matrices

    def get_skill_population_matrix(self) -> Any:
        return self._skill_population_matrix

    def calc_rca_matrix(self) -> NDArray[np.float64] | csr_array:
        """Computes the Revealed Comparative Advantage (RCA) matrix, where every row is a job advert (or more
        generically a skill group), every column is a skill and every element is RCA(job, skill).

        Returns:
            NDArray[np.float64]: 2D matrix of size (number of job adverts) x (number of unique skills), where
            each element is RCA(job, skill).
        """

        num_skills_in_jobs = xp.sum(self._skill_population_matrix, axis=1)[
            :, xp.newaxis
        ]

        nums_jobs_with_skill = xp.sum(self._skill_population_matrix, axis=0)
        num_skills = xp.sum(self._skill_population_matrix)

        self._rca_matrix = (self._skill_population_matrix / num_skills_in_jobs) / (
            nums_jobs_with_skill / num_skills
        )

    def get_rca_matrix(self):
        return self._rca_matrix

    def set_rca_matrix(self, rca_matrix):
        self._rca_matrix = rca_matrix

    def calc_skill_sim_matrix(self) -> NDArray[np.float64]:
        """Computes the Skill Similarity matrix (theta matrix) using the computed RCA matrix. If the RCA matrix
        is None then it will compute it as part of the execution of this method.

        Returns:
            NDArray[np.float64]: 2D matrix of size (number of unique skills) x (number of unique skills), where
            each element is skill_sim(skill_i, skill_j).
        """

        if self._rca_matrix is None:
            self.calc_rca_matrix()

        effective_use_matrix = None

        if self._use_sparse_matrices:
            effective_use_matrix = copy.deepcopy(self._rca_matrix)
            effective_use_matrix.data = (effective_use_matrix.data >= 1.0).astype(float)
        else:
            effective_use_matrix = xp.where(self._rca_matrix >= 1.0, 1.0, 0.0)

        skill_joint_freq_matrix = effective_use_matrix.T @ effective_use_matrix

        skill_effective_freq_vector = (
            skill_joint_freq_matrix.diagonal()
        )  # will be of type ndarray

        q1 = xp.tile(skill_effective_freq_vector, (len(skill_effective_freq_vector), 1))
        q2 = q1.T

        antecedent_matrix = xp.maximum(q1, q2)

        self._skill_sim_matrix = (
            skill_joint_freq_matrix / csr_array(antecedent_matrix)
            if self._use_sparse_matrices
            else skill_joint_freq_matrix / antecedent_matrix
        )

    def get_skill_sim_matrix(self):
        return self._skill_sim_matrix

    def set_skill_sim_matrix(self, skill_sim_matrix):
        self._skill_sim_matrix = skill_sim_matrix

    def skill_set_one_hot_vector(
        self, population_subset: MatrixSubsetIndexes
    ) -> NDArray[np.int64]:
        """Creates a one hot encoded vector (the length of the number of unique skills), where element at index i
        is 1 if it appears one or more time in the job population subset, and 0 if otherwise.

        Args:
            population_subset (MatrixSubsetIndexes): The job population subset (or job sample set).

        Returns:
            NDArray[np.int64]: 1D binary vector of size equal to the number of the number of unique skills.
        """

        skill_population_matrix_subset = self._skill_population_matrix[
            population_subset.indexes
        ]

        if self._use_sparse_matrices:
            skill_population_matrix_subset = skill_population_matrix_subset.toarray()

        return xp.clip(
            xp.sum(skill_population_matrix_subset, axis=0),
            None,
            1,
        )

    def skill_weight_vector(
        self, population_subset: MatrixSubsetIndexes
    ) -> NDArray[np.float64]:
        """Computes the skill set weight vector, where each element is the weight of a skill in a skill set (0
        when the skill is not present). This vector is of size of the number of unique skills.

        Args:
            population_subset (MatrixSubsetIndexes): The job population subset (or job sample set).

        Returns:
            _type_: 1D vector of size equal to the number of the number of unique skills.
        """

        if self._use_sparse_matrices:
            return (
                self.skill_set_one_hot_vector(population_subset)
                * self._rca_matrix.tocsr()[population_subset.indexes]
            ).sum(axis=0) / len(population_subset)

        return xp.sum(
            self.skill_set_one_hot_vector(population_subset)
            * self._rca_matrix[population_subset.indexes],
            axis=0,
        ) / len(population_subset)

    def get_skill_weight_components(
        self,
        population_subset_1: MatrixSubsetIndexes,
        population_subset_2: MatrixSubsetIndexes | None,
        custom_skill_weight_vector: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Calculates the skill weight vector for subset 1 and 2 individually. Then calculates a skill product
        matrix using the element wise dot product of the two weight vectors and a all-ones 2D matrix.

        Args:
            population_subset_1 (MatrixSubsetIndexes): The 1st job population subset (or job sample set).
            population_subset_2 (MatrixSubsetIndexes | None): The 2nd job population subset (or job sample set).
            custom_skill_weight_vector (NDArray[np.float64] | None, optional): A custom weight vector that will
            replace the weight vector of the 2nd job population subset. Defaults to None.

        Raises:
            Exception: If the method is called with subset 2 and custom weight vector 2 being None, then an
            error is thrown. One of those arguments is required to be of it's type hint.

        Returns:
            tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: A tuple containing the skill set
            weight vector of subset 1 and 2, and the skill product weight matrix.
        """

        if self._skill_sim_matrix is None:
            self.calc_skill_sim_matrix()

        if population_subset_2 is None and custom_skill_weight_vector is None:
            raise Exception(
                "A second matrix subset or custom skill weight vector needs to be provided. Both cannot be None."
            )

        # skills that aren't included in the skill set of subset 1 or 2 will have a weight of zero
        # when the element-wise dot product is calculated with the skill sim matrix this means
        # skills that weren't included in the subset will be zeroed out and not count towards the sum
        skill_set_weight_vector_1 = self.skill_weight_vector(population_subset_1)
        skill_set_weight_vector_2 = (
            self.skill_weight_vector(population_subset_2)
            if custom_skill_weight_vector is None
            else custom_skill_weight_vector
        )

        _, num_skills = self._skill_sim_matrix.shape
        skill_prod_weight_matrix = xp.ones((num_skills, num_skills))
        skill_prod_weight_matrix = (
            skill_set_weight_vector_1[:, xp.newaxis] * skill_prod_weight_matrix
        )
        skill_prod_weight_matrix = skill_set_weight_vector_2 * skill_prod_weight_matrix

        return (
            skill_set_weight_vector_1,
            skill_set_weight_vector_2,
            skill_prod_weight_matrix,
        )

    def skill_set_similarity(
        self,
        population_subset_1: MatrixSubsetIndexes,
        population_subset_2: MatrixSubsetIndexes | None,
        custom_skill_weight_vector: NDArray[np.float64] | None = None,
    ) -> np.float64:
        """Computes the Skill Set Similarity between job population subset 1 and 2 (alternatively this can be
            interpreted as the similarity score between skill set of the jobs of subset 1 and 2).

        Args:
            population_subset_1 (MatrixSubsetIndexes): The 1st job population subset (or job sample set).
            population_subset_2 (MatrixSubsetIndexes | None): The 2nd job population subset (or job sample set).
            custom_skill_weight_vector (NDArray[np.float64] | None, optional): A custom weight vector that will
            replace the weight vector of the 2nd job population subset. Defaults to None.

        Returns:
            np.float64: The final Skill Set Similarity score of job population subset 1 and 2 (or this can be
            interpreted as the similarity score between skill set of the jobs of subset 1 and 2).
        """

        (
            skill_set_weight_vector_1,
            skill_set_weight_vector_2,
            skill_prod_weight_matrix,
        ) = self.get_skill_weight_components(
            population_subset_1, population_subset_2, custom_skill_weight_vector
        )

        return np.float64(
            xp.sum(
                skill_set_weight_vector_2
                * (skill_set_weight_vector_1[:, xp.newaxis] * self._skill_sim_matrix)
            )
            / xp.sum(skill_prod_weight_matrix)
        )
