from dataclasses import dataclass

from entities import Skill


@dataclass
class Job:
    id: str
    name: str
    skills: set[Skill]
