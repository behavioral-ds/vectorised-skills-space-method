from dataclasses import dataclass

from tqdm.notebook import tqdm

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

    for job_index in skill_group.matrix.shape[0]:
        skills = []

        for skill_index in skill_group.matrix.shape[1]:
            if skill_group.matrix[job_index][skill_index] == 1:
                skills.append(skill_group.skill_names[skill_index])

        job_population.append(
            Job(job_index, skill_group.skill_group_names[job_index], skills)
        )

    return job_population


class SkillSimCalculatorBaseline(SkillSim):
    job_population: list[Job]

    def __init__(self, job_population: list[Job]):
        self.job_population = job_population

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

        return skill_pairwise_total / (
            skill1_total_effective_use
            if skill1_total_effective_use > skill2_total_effective_use
            else skill2_total_effective_use
        )

    def skill_weight(self, skill: Skill, skill_set: set[Skill]) -> float:
        total_rca = 0

        for job in self.job_population:
            if skill_set == job.skills:
                total_rca += self.rca(skill, job)

        return total_rca / len(self.job_population)

    def skill_set_similiarity(
        self, skill_set1: list[Skill], skill_set2: list[Skill]
    ) -> float:
        similarity_total = 0
        skill_weight_total = 0
        skill2_to_weight = (
            {}
        )  # memoise skill2 weights (skill1 weights will only be computed once)

        pbar = tqdm(total=len(skill_set1) * len(skill_set2))

        for skill1 in skill_set1:
            for skill2 in skill_set2:
                skill2_weight: float = None

                if skill2 in skill2_to_weight:
                    skill2_weight = skill2_to_weight[skill2]
                else:
                    skill2_weight = self.skill_weight(skill2, skill_set2)
                    skill2_to_weight[skill2] = skill2_weight

                skill_weight_product = (
                    self.skill_weight(skill1, skill_set1) * skill2_weight
                )

                similarity_total += skill_weight_product * self.skill_cooccurence(
                    skill1, skill2
                )
                skill_weight_total += skill_weight_product

                pbar.update(1)

        pbar.close()

        return similarity_total / skill_weight_total
