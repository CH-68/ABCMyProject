import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from src.tools import PolicySearchTool, UserDocSearchTool
from src.schemas import ComplianceReportSchema, SemanticFinding

@CrewBase
class ComplianceCrew:
    """Compliance Crew for verifying documents against policy using dynamic RAG tools."""
    agents_config = "../config/agents.yaml"
    tasks_config = "../config/tasks.yaml"

    def __init__(self, policy_vectorstore, user_vectorstore):
        # Instantiate custom tools with vectorstore references
        self.policy_tool = PolicySearchTool(vectorstore=policy_vectorstore)
        self.user_tool = UserDocSearchTool(vectorstore=user_vectorstore)

    @agent
    def semantic_evaluator(self) -> Agent:
        return Agent(
            config=self.agents_config["semantic_evaluator"],
            tools=[self.policy_tool, self.user_tool],
            verbose=True,
        )

    @task
    def semantic_evaluation_task(self) -> Task:
        return Task(
            config=self.tasks_config["semantic_evaluation_task"],
            agent=self.semantic_evaluator(),
            output_pydantic=SemanticFinding,
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )

    def verify_requirement(self, requirement: dict) -> SemanticFinding:
        """Kicks off the crew workflow for a single policy requirement."""
        # Handle requirement safely whether passed as a dict or object
        req_text = requirement.get("text") if isinstance(requirement, dict) else getattr(requirement, "text", str(requirement))
        
        my_crew = self.crew()
        result = my_crew.kickoff(
            inputs={
                "requirement": req_text
            }
        )
        
        # Returns the structured pydantic output from the task execution
        return result.pydantic