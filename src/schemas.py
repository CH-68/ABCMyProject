from typing import List, Literal
from pydantic import BaseModel, Field

Verification = Literal["passed", "deviated", "failed"]
IssueType = Literal["contradiction", "misalignment", "omission", "other"]

class SemanticFinding(BaseModel):
    requirement_id: str = Field(description="Requirement identifier (R-index).")
    verification: Verification = Field(description="Audit outcome.")
    issue_type: IssueType = Field(description="Nature of the semantic issue.")
    policy_citation: str = Field(description="Citation from policy document (Page, Section, Topic).")
    user_citation: str = Field(description="Citation from user document (Page, Section, Topic) or 'Not present'.")
    review_highlight: str = Field(description="Explanation of semantic differences in the exact words and deviation from policy requirements.")
    suggested_edits: str = Field(description="Recommended revision.")

class ComplianceSummary(BaseModel):
    total_requirements: int
    passed: int
    deviated: int
    failed: int

class ComplianceReportSchema(BaseModel):
    summary: ComplianceSummary | None = None
    findings: List[SemanticFinding]