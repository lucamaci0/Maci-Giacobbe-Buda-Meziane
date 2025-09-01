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
        """Agent that rewrites natural language math problems into formal math.

        Creates a math translation agent configured to convert natural language
        mathematical problems into formal mathematical expressions.

        Returns
        -------
        Agent
            A configured CrewAI agent specialized in mathematical translation.

        Examples
        --------
        >>> crew = Mathcrew()
        >>> agent = crew.math_translator()
        >>> print(type(agent))
        <class 'crewai.agent.Agent'>
        """
        return Agent(
            config=self.agents_config['math_translator'], # type: ignore[index]
            verbose=True
        )

    @agent
    def math_to_code_translator(self) -> Agent:
        """Agent that converts math expressions into runnable code.

        Creates a math-to-code translation agent configured to convert
        formal mathematical expressions into executable code.

        Returns
        -------
        Agent
            A configured CrewAI agent specialized in math-to-code translation.

        Examples
        --------
        >>> crew = Mathcrew()
        >>> agent = crew.math_to_code_translator()
        >>> print(type(agent))
        <class 'crewai.agent.Agent'>
        """
        return Agent(
            config=self.agents_config['math_to_code_translator'], # type: ignore[index]
            verbose=True
        )
        
    @agent
    def math_executor(self) -> Agent:
        """Agent that executes generated code using a code interpreter tool.

        Creates a math execution agent configured with the CodeInterpreterTool
        to execute generated mathematical code and return results.

        Returns
        -------
        Agent
            A configured CrewAI agent with code execution capabilities.

        Examples
        --------
        >>> crew = Mathcrew()
        >>> agent = crew.math_executor()
        >>> print(type(agent))
        <class 'crewai.agent.Agent'>
        """
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
        """Task that produces a formal math representation from a problem statement.

        Creates a math translation task configured to convert natural language
        mathematical problems into formal mathematical expressions.

        Returns
        -------
        Task
            A configured CrewAI task for mathematical translation operations.

        Examples
        --------
        >>> crew = Mathcrew()
        >>> task = crew.math_translation_task()
        >>> print(type(task))
        <class 'crewai.task.Task'>
        """
        return Task(
            config=self.tasks_config['math_translation_task'], # type: ignore[index]
        )

    @task
    def math_to_code_task(self) -> Task:
        """Task that generates code implementing the translated math.

        Creates a math-to-code task configured to convert formal mathematical
        expressions into executable code implementations.

        Returns
        -------
        Task
            A configured CrewAI task for math-to-code conversion operations.

        Examples
        --------
        >>> crew = Mathcrew()
        >>> task = crew.math_to_code_task()
        >>> print(type(task))
        <class 'crewai.task.Task'>
        """
        return Task(
            config=self.tasks_config['math_to_code_task'], # type: ignore[index]
        )
        
    @task
    def math_execution_task(self) -> Task:
        """Task that executes the generated code and returns the result.

        Creates a math execution task configured to execute generated
        mathematical code and return the calculated results.

        Returns
        -------
        Task
            A configured CrewAI task for mathematical code execution operations.

        Examples
        --------
        >>> crew = Mathcrew()
        >>> task = crew.math_execution_task()
        >>> print(type(task))
        <class 'crewai.task.Task'>
        """
        return Task(
            config=self.tasks_config['math_execution_task'], # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Create and return the math-oriented crew.

        Assembles and configures a CrewAI crew with math translation, code
        generation, and execution agents and tasks for mathematical problem solving.

        Returns
        -------
        Crew
            A configured CrewAI crew ready for mathematical problem solving tasks.

        Examples
        --------
        >>> crew_instance = Mathcrew()
        >>> crew = crew_instance.crew()
        >>> print(type(crew))
        <class 'crewai.crew.Crew'>
        """
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
