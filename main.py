import numpy as np

from SkillGroup import SkillGroup
from utils.get_random_skill_group import get_random_skill_group
from utils.MatrixSubsetIndexes import MatrixSubsetIndexes
from SkillSimCalculatorBaseline import SkillSimCalculatorBaseline, get_job_population
from SkillSimCalculatorV2 import SkillSimCalculatorV2
from utils.get_skill_group_matrix import get_skill_group_matrix


def main():
    skill_group = get_random_skill_group()
    job_population = get_job_population(skill_group)

    skill_sim_calc = SkillSimCalculatorBaseline(
        job_population, skills=skill_group.skill_names
    )
    skill_sim_calc_2 = SkillSimCalculatorV2(skill_group)

    matrix_subset_1 = MatrixSubsetIndexes((0, 4))
    matrix_subset_2 = MatrixSubsetIndexes([len(job_population) - i for i in range(1, 6)])
    
    # print(np.array_equal(get_skill_group_matrix(job_population[:11], skill_group.skill_names), skill_group.matrix[:11]))
    # print(np.array_equal(get_skill_group_matrix(job_population[-10:], skill_group.skill_names), skill_group.matrix[-10:]))
    
    print("baseline:", skill_sim_calc.skill_set_similiarity(job_population[:5], job_population[-5:]))
    print("v2:", skill_sim_calc_2.skill_set_similiarity(matrix_subset_1, matrix_subset_2))

if __name__ == "__main__":
    main()
