from dataclasses import dataclass
from typing import Optional


@dataclass
class SkillGroup:
    id: str
    properties: Optional[dict] = None
