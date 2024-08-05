from entities import SkillGroup
from entities import Job, Skill


def get_job_population(skill_group: SkillGroup) -> list[Job]:
    job_population = []

    num_jobs, num_skills = skill_group.matrix.shape

    for job_index in range(num_jobs):
        skills = []

        for skill_index in range(num_skills):
            if skill_group.matrix[job_index][skill_index] == 1:
                skills.append(Skill(name=skill_group.skill_names[skill_index]))

        job_population.append(
            Job(job_index, skill_group.skill_group_names[job_index], skills)
        )

    return job_population
