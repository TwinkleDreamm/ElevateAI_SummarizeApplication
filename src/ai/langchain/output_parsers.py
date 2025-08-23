from typing import List, Dict, Any

try:
    from langchain.output_parsers import PydanticOutputParser
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover
    PydanticOutputParser = None  # type: ignore
    BaseModel = object  # type: ignore
    def Field(*args, **kwargs):  # type: ignore
        return None


class SummaryOutput(BaseModel):  # type: ignore[misc]
    summary: str = Field(description="Main summary of the content")  # type: ignore[assignment]
    key_points: List[str] = Field(default_factory=list, description="Key points extracted")  # type: ignore[assignment]
    confidence: float = Field(default=0.8, description="Confidence 0-1")  # type: ignore[assignment]
    topics: List[str] = Field(default_factory=list, description="Main topics")  # type: ignore[assignment]


class QAOutput(BaseModel):  # type: ignore[misc]
    answer: str = Field(description="Direct answer")  # type: ignore[assignment]
    confidence: float = Field(default=0.8, description="Confidence 0-1")  # type: ignore[assignment]
    sources: List[str] = Field(default_factory=list, description="Sources used")  # type: ignore[assignment]
    reasoning: str | None = Field(default=None, description="Optional reasoning")  # type: ignore[assignment]
    citations: List[Dict[str, str]] = Field(default_factory=list, description="Specific citations with page/line info")  # type: ignore[assignment]


class LangchainOutputParser:
    """Structured output parsing helpers (optional dependency)."""

    def __init__(self) -> None:
        if PydanticOutputParser is None:
            self.summary_parser = None
            self.qa_parser = None
        else:
            self.summary_parser = PydanticOutputParser(pydantic_object=SummaryOutput)
            self.qa_parser = PydanticOutputParser(pydantic_object=QAOutput)

    def parse_summary(self, response_text: str) -> SummaryOutput:
        if self.summary_parser is None:
            # Fallback to minimal structure
            return SummaryOutput(summary=response_text)  # type: ignore[call-arg]
        try:
            return self.summary_parser.parse(response_text)
        except Exception:
            return SummaryOutput(summary=response_text)  # type: ignore[call-arg]

    def parse_qa(self, response_text: str) -> QAOutput:
        if self.qa_parser is None:
            return QAOutput(answer=response_text)  # type: ignore[call-arg]
        try:
            return self.qa_parser.parse(response_text)
        except Exception:
            return QAOutput(answer=response_text)  # type: ignore[call-arg]

    def extract_citations(self, answer: str, sources: List[str]) -> List[Dict[str, str]]:
        """Extract citations from answer text and map to sources."""
        citations = []
        for i, source in enumerate(sources, 1):
            # Look for citation patterns like [1], [source], etc.
            if f"[{i}]" in answer or f"[{source}]" in answer:
                citations.append({
                    "source": source,
                    "reference": f"[{i}]",
                    "type": "explicit"
                })
        return citations


