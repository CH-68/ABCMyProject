from typing import Dict, List, Literal
from pydantic import BaseModel, Field

# ----------------------------
# Core data models
# ----------------------------

class Issue(BaseModel):
    """One candidate compliance issue produced by the generator."""
    id: str = Field(default="")
    title: str = Field(default="")
    summary: str = Field(default="")
    target_rule: str = Field(default="")
    evidence_pointers: List[str] = Field(default_factory=list)


class IssueVerification(BaseModel):
    """One independent verifier result for one issue."""
    issue_id: str = Field(default="")
    verdict: Literal["pass", "fail", "insufficient"] = Field(default="insufficient")
    rationale: str = Field(default="")
    evidence: List[str] = Field(default_factory=list)
    suggested_fix: str = Field(default="")

 
class ComplianceReportSchema(BaseModel):
    """Final plain dictionary schema returned by run_verification()."""
    verdict: Literal["fully_cleared", "partially_cleared", "failed"]
    summary: str = Field(default="")
    issue_inventory: List[Issue] = Field(default_factory=list)
    issue_results: List[IssueVerification] = Field(default_factory=list)
    evidence_map: Dict[str, List[str]] = Field(default_factory=dict)
    requirements_covered: List[str] = Field(default_factory=list)
    requirements_missing: List[str] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)
    explanation: str = Field(default="")


