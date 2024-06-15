from dataclasses import dataclass
from itertools import chain

from tqdm import tqdm

from SkillSim import SkillSim
from SkillGroup import SkillGroup


@dataclass
class Skill:
    # skill is a class so other properties can be added later
    name: str

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, value: object) -> bool:
        return self.name == value.name


@dataclass
class Job:
    id: str
    name: str
    skills: set[Skill]


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


class SkillSimCalculatorBaseline(SkillSim):
    job_population: list[Job]
    skills: list[Skill] | None

    def __init__(self, job_population: list[Job], skills: list[str] | None = None):
        self.job_population = job_population
        self.skills = (
            [Skill(name=skill) for skill in skills] if skills is not None else None
        )

    def rca(self, skill: Skill, job: Job) -> float:
        # RCA numerator
        is_skill_in_job = 1 if skill in job.skills else 0

        if is_skill_in_job == 0:
            return 0

        num_skills_in_job = len(job.skills)

        # RCA denominator
        num_jobs_with_skill = 0
        num_skills = 0

        for current_job in self.job_population:
            for current_skill in current_job.skills:
                num_skills += 1

                if skill == current_skill:
                    num_jobs_with_skill += 1

        return (is_skill_in_job / num_skills_in_job) / (
            num_jobs_with_skill / num_skills
        )

    def skill_cooccurence(self, skill1: Skill, skill2: Skill) -> float:
        skill_pairwise_total = 0
        skill1_total_effective_use = 0
        skill2_total_effective_use = 0

        for job in self.job_population:
            is_skill1_effective = 1 if self.rca(skill1, job) >= 1 else 0
            is_skill2_effective = 1 if self.rca(skill2, job) >= 1 else 0

            skill_pairwise_total += is_skill1_effective * is_skill2_effective
            skill1_total_effective_use += is_skill1_effective
            skill2_total_effective_use += is_skill2_effective

        if skill1_total_effective_use == 0 and skill2_total_effective_use == 0:
            return 0

        return skill_pairwise_total / (
            skill1_total_effective_use
            if skill1_total_effective_use > skill2_total_effective_use
            else skill2_total_effective_use
        )

    def skill_weight(self, skill: Skill, jobs: list[Job]) -> float:
        total_rca = 0

        for job in jobs:
            total_rca += self.rca(skill, job)

        return total_rca / len(jobs)

    def skill_set_similiarity(self, jobs1: list[Job], jobs2: list[Job]) -> float:
        weighted_skills_coocurrence = 0
        weighted_skills_total = 0

        skill_set1 = set(chain.from_iterable([list(job.skills) for job in jobs1]))
        skill_set2 = set(chain.from_iterable([list(job.skills) for job in jobs2]))

        pbar = tqdm(total=len(skill_set1) * len(skill_set2))

        for skill1 in skill_set1:
            job1_skill_weight = self.skill_weight(skill1, jobs1)

            for skill2 in skill_set2:
                job2_skill_weight = self.skill_weight(skill2, jobs2)
                cooccurrence_val = self.skill_cooccurence(skill1, skill2)

                weighted_skills = job1_skill_weight * job2_skill_weight
                weighted_skills_total += weighted_skills

                weighted_skills_coocurrence += weighted_skills * cooccurrence_val

                pbar.update(1)

        pbar.close()

        return weighted_skills_coocurrence / weighted_skills_total


def dep_skill_set_similiarity(
    self, skill_sets_1: list[Job], skill_sets_2: list[Job]
) -> float:
    pbar = tqdm(
        total=len(list(chain.from_iterable([list(job.skills) for job in skill_sets_1])))
        * len(list(chain.from_iterable([list(job.skills) for job in skill_sets_2])))
    )

    weighted_skills_coocurrence = 0
    weighted_skills_total = 0

    for job1 in skill_sets_1:
        for job1_skill in job1.skills:
            job1_skill_weight = self.skill_weight(job1_skill, skill_sets_1)

            for job2 in skill_sets_2:
                for job2_skill in job2.skills:
                    job2_skill_weight = self.skill_weight(job2_skill, skill_sets_2)
                    cooccurrence_val = self.skill_cooccurence(job1_skill, job2_skill)

                    weighted_skills = job1_skill_weight * job2_skill_weight
                    weighted_skills_total += weighted_skills

                    weighted_skills_coocurrence += weighted_skills * cooccurrence_val

                    pbar.update(1)

    pbar.close()

    return weighted_skills_coocurrence / weighted_skills_total
