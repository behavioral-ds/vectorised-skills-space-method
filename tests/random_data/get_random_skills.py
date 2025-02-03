import random
from uuid import uuid4

from entities.SkillGroup import SkillGroup

def get_random_occ_to_skills(
    num_occupations=15, num_skills=10
) -> dict[str, list[list[str]]]:
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

    return rand_occ_to_jobs_skills

def add_metadata_to_occ_skills(occ_to_skills: dict[str, list[list[str]]]) -> dict[str, SkillGroup]:
    return {occupation: {"name": occupation, "skill_sets": [{"id": str(uuid4()), "skills": skill_set} for skill_set in skill_sets]} for (occupation, skill_sets) in occ_to_skills.items()}
