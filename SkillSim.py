from abc import ABC, abstractmethod

from SkillGroup import SkillGroup
from utils.MatrixSubsetIndexes import MatrixSubsetIndexes


class SkillSim(ABC):
    @abstractmethod
    def skill_set_similiarity(
        self,
        matrix_subset1: MatrixSubsetIndexes,
        matrix_subset2: MatrixSubsetIndexes,
    ) -> float:
        pass
