from abc import ABC, abstractmethod

from utils.MatrixSubsetIndexes import MatrixSubsetIndexes

import numpy as np


class SkillSim(ABC):
    @abstractmethod
    def skill_set_similarity(
        self,
        matrix_subset_1: MatrixSubsetIndexes,
        matrix_subset_2: MatrixSubsetIndexes,
    ) -> float | np.float64:
        pass
