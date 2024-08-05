from dataclasses import dataclass


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
