"""Crew definition for handling math-related queries.

Defines agents and tasks to translate math problems to code and execute them
with an interpreter tool.
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import CodeInterpreterTool
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class Mathcrew():
    """Crew that translates math problems to executable code and runs it."""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def math_translator(self) -> Agent:
        """Agent that rewrites natural language math problems into formal math."""
        return Agent(
            config=self.agents_config['math_translator'], # type: ignore[index]
            verbose=True
        )

    @agent
    def math_to_code_translator(self) -> Agent:
        """Agent that converts math expressions into runnable code."""
        return Agent(
            config=self.agents_config['math_to_code_translator'], # type: ignore[index]
            verbose=True
        )
        
    @agent
    def math_executor(self) -> Agent:
        """Agent that executes generated code using a code interpreter tool."""
        return Agent(
            config=self.agents_config['math_executor'], # type: ignore[index]
            verbose=True,
            tools=[CodeInterpreterTool()]
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def math_translation_task(self) -> Task:
        """Task that produces a formal math representation from a problem statement."""
        return Task(
            config=self.tasks_config['math_translation_task'], # type: ignore[index]
        )

    @task
    def math_to_code_task(self) -> Task:
        """Task that generates code implementing the translated math."""
        return Task(
            config=self.tasks_config['math_to_code_task'], # type: ignore[index]
        )
        
    @task
    def math_execution_task(self) -> Task:
        """Task that executes the generated code and returns the result."""
        return Task(
            config=self.tasks_config['math_execution_task'], # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Create and return the math-oriented crew."""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
