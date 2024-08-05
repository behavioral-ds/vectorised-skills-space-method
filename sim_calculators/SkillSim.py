from abc import ABC, abstractmethod

from utils.MatrixSubsetIndexes import MatrixSubsetIndexes


class SkillSim(ABC):
    @abstractmethod
    def skill_set_similarity(
        self,
        matrix_subset_1: MatrixSubsetIndexes,
        matrix_subset_2: MatrixSubsetIndexes,
    ) -> float:
        pass
