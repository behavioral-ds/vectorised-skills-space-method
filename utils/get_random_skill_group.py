import random

from SkillGroup import SkillGroup


def get_random_skill_group():
    num_occupations = 15
    num_skills = 10

    occupations = [f"occupation-{i}" for i in range(1, num_occupations + 1)]
    skills = [f"skill-{i}" for i in range(1, num_skills + 1)]

    rand_occ_to_jobs_skills = {}

    def get_random_skills() -> list[str]:
        return [
            skills[i]
            for i in set(
                [
                    random.randint(0, num_skills - 1)
                    for _ in range(random.randint(0, int((num_skills - 1) * 0.5)))
                ]
            )
        ]

    max_num_jobs = 10

    for occupation in occupations:
        jobs_skill_sets = [
            get_random_skills() for _ in range(random.randint(1, max_num_jobs + 1))
        ]

        rand_occ_to_jobs_skills[occupation] = jobs_skill_sets

    skill_group = SkillGroup(rand_occ_to_jobs_skills)

    return skill_group
