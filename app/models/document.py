from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class Document:

    text: str
    file_name: str
    folder: str
    page_number: int

    metadata: Dict[str, Any] = field(default_factory=dict)

    embedding: list[float] | None = None