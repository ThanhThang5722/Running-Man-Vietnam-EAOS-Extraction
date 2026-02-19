from pydantic import BaseModel, field_validator
from typing import List

class EAOSLabel(BaseModel):
    """Single EAOS (Entity-Aspect-Opinion-Sentiment) label"""
    entity: str
    aspect: str
    opinion: str
    sentiment: str

class ModelInput(BaseModel):
    text: str

class ModelOutput(BaseModel):
    results: List[EAOSLabel]

    @field_validator('results')
    def remove_duplicates(cls, v):
        unique_results = []
        seen = set()
        for item in v:
            key = str(item)
            if key not in seen:
                unique_results.append(item)
                seen.add(key)
        return unique_results