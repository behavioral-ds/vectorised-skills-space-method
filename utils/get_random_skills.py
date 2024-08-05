import random
import copy
from uuid import uuid4

from entities import SkillPopulation
from entities import SkillGroup


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


def get_random_skill_group(
    num_occupations=15,
    num_skills=10,
) -> SkillGroup:
    return SkillGroup(get_random_occ_to_skills(num_occupations, num_skills))


def get_random_skill_population(
    include_rand_props: bool, rand_occ_to_jobs_skills: dict[str, list[list[str]]]
) -> SkillPopulation:
    occ_to_jobs_skills = copy.deepcopy(rand_occ_to_jobs_skills)

    # code to add random props to random skill groups and skill sets
    if include_rand_props:
        # random occupations to add props to random jobs (skill sets)
        random_occupations = random.sample(
            list(occ_to_jobs_skills.keys()),
            k=random.randint(0, (len(occ_to_jobs_skills) - 1) // 2),
        )

        for i, occupation in enumerate(random_occupations):
            random_skill_set_indexes = list(
                set(
                    [
                        random.randint(
                            0, (len(occ_to_jobs_skills[occupation]) - 1) // 2
                        )
                        for _ in range(
                            random.randint(
                                0, (len(occ_to_jobs_skills[occupation]) - 1) // 2
                            )
                        )
                    ]
                )
            )

            for i, skill_set_index in enumerate(random_skill_set_indexes):
                skill_set = occ_to_jobs_skills[occupation][skill_set_index]

                occ_to_jobs_skills[occupation][skill_set_index] = {
                    "id": str(uuid4()),
                    "id_source": "test",
                    "prop_a": i,
                    "prop_b": i + 1,
                    "skills": skill_set,
                }

        # random occupations to add props to occupation/job title
        random_occupations = random.sample(
            list(occ_to_jobs_skills.keys()),
            k=random.randint(0, len(occ_to_jobs_skills) - 1),
        )

        for i, occupation in enumerate(random_occupations):
            skill_sets = occ_to_jobs_skills[occupation]

            occ_to_jobs_skills[occupation] = {
                "prop1": i,
                "prop2": i + 1,
                "skill_sets": skill_sets,
            }

    return SkillPopulation(occ_to_jobs_skills)
