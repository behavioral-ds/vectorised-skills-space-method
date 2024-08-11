import time
import argparse
import csv
import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add parent directory to the sys.path
sys.path.insert(0, parent_dir)

from tqdm import tqdm

from tests import get_random_skill_group
from utils import get_job_population
from sim_calculators import SkillSimCalculatorBaseline

num_occupations = 15
num_skills = 5

parser = argparse.ArgumentParser()

parser.add_argument(
    "-s",
    "--save-path",
    type=str,
    required=True,
    help="File path to save CSV result to.",
)

args = parser.parse_args()

save_path = args.save_path

results = [
    [
        "Test Iteration",
        "Number of Unique Occupations",
        "Number of Jobs",
        "Number of Unique Skills",
        "Mean Execution Time",
    ]
]

for pop_magnitude in tqdm(range(1, 7)):
    current_num_occupations = num_occupations * pop_magnitude
    current_num_skills = num_skills * pop_magnitude

    skill_group = get_random_skill_group(current_num_occupations, current_num_skills)
    job_population = get_job_population(skill_group)

    mean_time = 0

    for _ in range(3):
        skill_sim_calc = SkillSimCalculatorBaseline(
            job_population, skills=skill_group.skill_names
        )

        half_index = len(job_population) // 2

        start = time.time()
        baseline_result = skill_sim_calc.skill_set_similarity(
            job_population[0 : half_index - 1],
            [job_population[i] for i in range(half_index + 1, len(job_population))],
        )
        end = time.time()
        baseline_time = round((end - start) * 1000, 2)

        mean_time += baseline_time

    mean_time = round(mean_time / 3, 2)
    results.append(
        [
            pop_magnitude,
            current_num_occupations,
            len(job_population),
            current_num_skills,
            mean_time,
        ]
    )

with open(save_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(results)
