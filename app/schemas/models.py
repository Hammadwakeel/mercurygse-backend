from pydantic import BaseModel, Field
from typing import List, Optional


class RouterOutput(BaseModel):
    route: str
    contains_visual: bool
    visual_types: List[str]
    reason: str = Field(..., min_length=8)
    confidence: float = Field(..., ge=0.0, le=1.0)


class KeyComponent(BaseModel):
    name: str
    description: str
    extraction_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class DiagramExtraction(BaseModel):
    schema_id: str = Field("diagram_v1")
    pdf_page: int
    printed_page: Optional[str]
    title: str
    category: str
    summary: str
    key_components: List[KeyComponent] = Field(default_factory=list)
    relationships: str
    raw_text: str
    extraction_confidence: float = Field(..., ge=0.0, le=1.0)


class SimpleExtraction(BaseModel):
    schema_id: str = Field("simple_v1")
    pdf_page: int
    printed_page: Optional[str]
    topic: str
    summary: str
    content_markdown: str
    important_dates_or_entities: List[str] = Field(default_factory=list)
    extraction_confidence: float = Field(..., ge=0.0, le=1.0)
