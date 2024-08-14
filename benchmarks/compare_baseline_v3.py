import sys
import os
import time
import argparse
import csv
import json
import random

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add parent directory to the sys.path
sys.path.insert(0, parent_dir)

from tqdm import tqdm

from entities import SkillPopulationDeprecated, SkillPopulation
from tests import get_random_occ_to_skills
from utils import get_job_population, MatrixSubsetIndexes
from sim_calculators import (
    SkillSimCalculatorBaseline,
    SkillSimCalculatorV2,
    SkillSimCalculatorV3,
)

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

parser.add_argument(
    "-r",
    "--create-random",
    action="store_true",
    help="Create random data if true, read data from ../data.json if false.",
)

args = parser.parse_args()

save_path = args.save_path
is_create_random_data = args.create_random

results = [
    [
        "Test Iteration",
        "Number of Unique Occupations",
        "Number of Jobs",
        "Number of Unique Skills",
        "Baseline MET",
        "V3 MET",
    ]
]

num_occupations = 15
num_skills = 5

test_occ_to_skills = []

if not is_create_random_data:
    with open("../data/data.json", "r") as f:
        test_occ_to_skills = json.load(f)

for test_num in tqdm(range(1, 7)):
    current_num_occupations = num_occupations * test_num
    current_num_skills = num_skills * test_num

    occ_to_skills = None

    if is_create_random_data:
        occ_to_skills = get_random_occ_to_skills(
            current_num_occupations, current_num_skills
        )

        test_occ_to_skills.append(occ_to_skills)
    else:
        occ_to_skills = test_occ_to_skills[test_num - 1]

    skill_group = SkillPopulationDeprecated(occ_to_skills)
    skill_population = SkillPopulation(skill_group_skills=occ_to_skills)

    job_population = get_job_population(skill_group)

    print(
        f"Test {test_num}: Occupations = {num_occupations * test_num}, Jobs = {len(job_population)}, Skills = {num_skills * test_num}"
    )

    skill_sim_calc_baseline = SkillSimCalculatorBaseline(
        job_population, skills=skill_group.skill_names
    )
    skill_sim_calc_v2 = SkillSimCalculatorV2(skill_group)
    skill_sim_calc_v3 = SkillSimCalculatorV3(skill_population)

    start_index = random.randint(0, len(job_population) - 2)
    end_index = random.randint(start_index + 1, len(job_population) - 1)

    matrix_subset_1 = MatrixSubsetIndexes((start_index, end_index))
    matrix_subset_2 = MatrixSubsetIndexes(
        list(
            set(
                [
                    random.randint(0, len(job_population) - 1)
                    for _ in range(random.randint(1, len(job_population)))
                ]
            )
        )
    )

    baseline_met = 0
    v3_met = 0

    for _ in range(3):
        start = time.time()
        baseline_result = skill_sim_calc_baseline.skill_set_similarity(
            [job_population[i] for i in matrix_subset_1],
            [job_population[i] for i in matrix_subset_2],
        )
        end = time.time()
        baseline_met += end - start

        start = time.time()
        v3_result = skill_sim_calc_v3.skill_set_similarity(
            matrix_subset_1, matrix_subset_2
        )
        end = time.time()
        v3_met += end - start

    baseline_met = round((baseline_met * 1000) / 3, 2)
    v3_met = round((v3_met * 1000) / 3, 2)

    results.append(
        [
            test_num,
            current_num_occupations,
            len(job_population),
            current_num_skills,
            baseline_met,
            v3_met,
        ]
    )

with open("../data/data.json", "w") as f:
    json.dump(test_occ_to_skills, f, indent=2)

with open(save_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(results)
