from SkillGroup import SkillGroup
from utils.get_random_skill_group import get_random_skill_group
from utils.MatrixSubsetIndexes import MatrixSubsetIndexes
from SkillSimCalculatorBaseline import SkillSimCalculatorBaseline, get_job_population
from SkillSimCalculatorV2 import SkillSimCalculatorV2


def main():
    skill_group = get_random_skill_group()
    job_population = get_job_population(skill_group)

    skill_sim_calc = SkillSimCalculatorBaseline(
        job_population, skills=skill_group.skill_names
    )
    cooccurrence_matrix_1 = skill_sim_calc.cooccurrence_matrix()
    print(cooccurrence_matrix_1.shape)
    print(cooccurrence_matrix_1)

    print()

    skill_sim_calc_2 = SkillSimCalculatorV2(skill_group)
    cooccurrence_matrix_2 = skill_sim_calc_2.cooccurrence_matrix()
    print(cooccurrence_matrix_2.shape)
    print(cooccurrence_matrix_2)

    # print("Number of jobs:", len(job_population))

    # random_jobs_1 = [
    #     random.choice(job_population) for _ in range(len(job_population) // 6)
    # ]
    # random_jobs_2 = [
    #     random.choice(job_population) for _ in range(len(job_population) // 3)
    # ]

    # similarity_score = skill_sim_calc.skill_set_similiarity(
    #     random_jobs_1, random_jobs_2
    # )

    # print("Skill Set Similarity Score:", similarity_score)


if __name__ == "__main__":
    main()
