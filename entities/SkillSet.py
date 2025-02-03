from typing import TypedDict, NotRequired, Any

SkillSet = TypedDict(
    "SkillSet",
    {
        "id": str,
        "skills": list[str],
        "id_source": NotRequired[str],
        "properties": NotRequired[dict[str, Any]],
    },
)
