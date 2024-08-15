from dataclasses import dataclass
from typing import Optional


@dataclass
class SkillSetMetadata:
    id: Optional[str] = None
    id_source: Optional[str] = None
    properties: Optional[dict] = None
