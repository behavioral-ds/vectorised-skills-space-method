# Vectorised Skills Space Method (VSSM)

#### An Implementation of Skill Set Similarity using Matrix Operations

## What is this Project?

The Skills Space Method developed in [“Skill-driven Recommendations for Job Transition Pathways” (Dawson, Williams & Rizoiu 2021)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0254722) is a novel approach to measuring the distance between sets of skills. Sets of skills in the context of this report are only associated with occupations (job titles), but can be generalised to other groups such as company or industry.

[(Dawson, Williams & Rizoiu 2021)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0254722) provide the mathematical foundation for computing the Skill Set Similarity between two sets. However, practically implementing the functions and computations is not documented. A naive approach, that mimics the exact mathematical outline (e.g. using for loops to sum, lazy functions) is not feasible on large datasets due to poor execution performance.

As such, this report outlines and explains a method for implementing the Skills Space Method using matrix operations. The vectorisation of this method enables faster execution through SIMD or parallel computing (depending on the platform) and memoisation of necessary, static values in matrix form.

## Project Dependencies and Setup

### Dependencies

- Python > 3.11
- Packages in platform specific requirements:
  - `nvidia_requirements.txt`: If you have an Nvidia GPU available and wish to utilise it for faster processing.
  - `mac_requirements`: If you are using a Mac and wish to utilise it's GPU it for faster processing.
  - `requirements.txt`: If you don't have an Nvidia or Apple GPU available or wish to only use CPU processing.

> [!IMPORTANT]
> To successfully install the required packages in `nvidia_requirements.txt` (specifically `cupy`), you may need to install other dependencies such as NVIDIA CUDA GPU and CUDA Toolkit. To determine what you need to install prior to installing `nvidia_requirements.txt` read through the [CuPy Installation Guide](https://docs.cupy.dev/en/stable/install.html).

### Installation via PIP

> [!NOTE]  
> Currently this package is not available on the Python Package Index (PIP). However, there are plans to make it available there in the future. So for now, this package has to be manually cloned into your project, and installed from source.

### Running from Source

```bash
git init
git clone <github-link>
cd skill-similarity
```

If you wish to install `nvidia_requirements` read the callout in [the Dependencies section](#dependencies).

```bash
pip install -r <platform-option>_requirements.txt
```

## Example VSSM Usage

### Setting the Path and Imports

If you are running this package from source, you must set the path of your local version of the repository as shown below:

```Python
import sys
sys.path.insert(1, "./skill_similarity")
```

Then import the `SkillPopulation` and `SkillCalculatorV3` classes:

```Python
from skill_similarity.tests.get_random_skills.get_random_occ_to_skills import get_random_occ_to_skills # only required to generate random data, omit if you have your own dataset
from skill_similarity.entities import SkillPopulation, SkillGroup
from skill_similarity.sim_calculators import SkillSimCalculatorV3
```

### Creating a Skill Population

Create some fake, random jobs and skills data:

```Python
job_skills = get_random_occ_to_skills(num_occupations=100, num_skills=70)
```

The `job_skills` variable returns a `dict[str, list[list[str]]]`, where the key is a occupation/job title (e.g. Junior Accountant, Senior Software Engineer), and the value is a list of list of skill names. Where each `list[str]` represents an instance of an occupation (e.g. a job posting) and it's skills.

Now that there is job-skill data, in the correct format, we must create a `SkillPopulation` using it. The `SkillPopulation` is necessary as it handles:

- Removing infrequent skills (skills that occur less than the `threshold` argument which is set to a default of 5).
- Creating a binary presence job-skill matrix.
- Creating an ordered list of skill_group_names (in this case job titles) and skill_names that can be associated with rows and columns of the skill population matrix respectively.
- Saving and loading the SkillPopulation to disk efficiently.

For a relatively small dataset (as is the case in this example) this can be done by simply by creating an instance of the `SkillPopulation`:

```Python
skill_population = SkillPopulation(job_skills)
```

However, for large datasets `skill_population` will use a lot of memory and possibly won't be released for a long time. In this case you should use a context manager to instantiate a `SkillPopulation`, save it if required and pass it to the skill calculator:

```Python
with SkillPopulation(skill_group_skills=job_skills) as skill_population:
    skill_population.save(file_path="../data", output_name="skill_population")
    skill_calc = SkillSimCalculatorV3(skill_population)
```

Afterwhich if you try to access the fields within `skill_population` (outside the context manager), you will get `None`.

### Calculating Skill Set Similarity

#### Getting Two Skill Population Subsets

The `get_matrix_subset_by_sg` method takes a callback function for filtering the `skill_population` matrix by skill_group and returns an object of `MatrixSubsetIndexes`, which is essentially a wrapper for the indexes of the matrix which were selected from the filter.

```Python
def filter_by_job_name(job_name: str, skill_group: SkillGroup) -> bool:
    return skill_group.id.lower() == job_name

soft_dev_subset = skill_population.get_matrix_subset_by_sg(
    lambda skill_group: SkillGroup: filter_by_job_name("Software Developer", skill_group)
)
data_sci_subset = skill_population.get_matrix_subset_by_sg(
    lambda skill_group: SkillGroup: filter_by_job_name("Data Scientist", skill_group)
)
```

#### Precomputation of the RCA and Skill Similarity Matrices

As mentioned in the implementation report, the RCA and Skill Similarity matrices are precomputed as this can be done very efficiently using matrix operations and enables memoisation for even faster subsequent computation.

Precompute these matrices by running their respective methods:

```Python
skill_calc.calc_rca_matrix()
skill_calc.calc_skill_sim_matrix()
```

#### Calculate Skill Set Similarity

The last step is to run the `skill_set_similarity` method, which takes two `MatrixSubsetIndexes` objects and returns a `np.float64` which is the skill set similarity value:

```Python
sss = skill_calc.skill_set_similarity(soft_dev_subset, data_sci_subset)
print("Skill Set Similarity =", sss)
```

## Project Structure

The project structure of VSSM is as follows:

- root: Contains `main.py`, requirements files and other configuration files.
    - entities: Contains classes or dataclasses that are commonly used throughout different modules. For example `SkillPopulation` and `Job`.
    - sim_calculators: Contains the different versions of the Skill Similarity calculators as well as the abstract class all are based off.
    - tests: Contains the unittests for this project.
        - random: Contains functions to generate random data for the tests.
    - utils: Contains various utility functions such as `gzip_directory` or `TwoDimDict`.

## Testing

- Run all tests: `python -m unittest discover -s tests -p "*_test.py"`
- Run a specific test: `python -m unittest tests.<test-module>.<test-class>.<test-method>`
