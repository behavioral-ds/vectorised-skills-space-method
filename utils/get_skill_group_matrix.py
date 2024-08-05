from typing import Any

import numpy as np

from sim_calculators.entities import Job


def get_skill_group_matrix(
    job_population: list[Job], skill_names: list[str]
) -> np.ndarray[Any, np.dtype[np.int8]]:
    sg_matrix = np.zeros((len(job_population), len(skill_names)))

    for row_index, job in enumerate(job_population):
        for skill in job.skills:
            col_index = skill_names.index(skill.name)
            sg_matrix[row_index][col_index] = 1

    return sg_matrix
