from typing import TypedDict, NotRequired, Any

from entities.SkillSet import SkillSet

SkillGroup = TypedDict(
    "SkillGroup", { "name": str, "skill_sets": list[SkillSet], "properties": NotRequired[dict[str, Any]] }
)
