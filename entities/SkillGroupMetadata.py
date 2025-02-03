from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class SkillGroupMetadata:
    id: str
    properties: Optional[dict[str, Any]] = None
