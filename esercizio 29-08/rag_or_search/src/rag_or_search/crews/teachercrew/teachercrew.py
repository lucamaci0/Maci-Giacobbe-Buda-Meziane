"""Crew definition for generating outlines and explanatory documents."""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class Teachercrew():
    """Crew that outlines topics and writes explanatory documents."""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def outline_generator(self) -> Agent:
        """Agent that creates a structured outline for the topic.

        Creates an outline generation agent configured to analyze content
        and generate structured outlines for educational purposes.

        Returns
        -------
        Agent
            A configured CrewAI agent specialized in outline generation.

        Examples
        --------
        >>> crew = Teachercrew()
        >>> agent = crew.outline_generator()
        >>> print(type(agent))
        <class 'crewai.agent.Agent'>
        """
        return Agent(
            config=self.agents_config['outline_generator'], # type: ignore[index]
            verbose=True
        )

    @agent
    def document_writer(self) -> Agent:
        """Agent that drafts an explanatory document from the outline.

        Creates a document writing agent configured to transform structured
        outlines into comprehensive explanatory documents.

        Returns
        -------
        Agent
            A configured CrewAI agent specialized in document writing.

        Examples
        --------
        >>> crew = Teachercrew()
        >>> agent = crew.document_writer()
        >>> print(type(agent))
        <class 'crewai.agent.Agent'>
        """
        return Agent(
            config=self.agents_config['document_writer'], # type: ignore[index]
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def outline_task(self) -> Task:
        """Task that produces an outline for the requested topic.

        Creates an outline generation task configured to analyze content
        and generate structured outlines for educational materials.

        Returns
        -------
        Task
            A configured CrewAI task for outline generation operations.

        Examples
        --------
        >>> crew = Teachercrew()
        >>> task = crew.outline_task()
        >>> print(type(task))
        <class 'crewai.task.Task'>
        """
        return Task(
            config=self.tasks_config['outline_task'], # type: ignore[index]
        )

    @task
    def writing_task(self) -> Task:
        """Task that writes an explanatory document based on the outline.

        Creates a document writing task configured to transform structured
        outlines into comprehensive explanatory documents and save them to a report file.

        Returns
        -------
        Task
            A configured CrewAI task for document writing operations.

        Examples
        --------
        >>> crew = Teachercrew()
        >>> task = crew.writing_task()
        >>> print(type(task))
        <class 'crewai.task.Task'>
        """
        return Task(
            config=self.tasks_config['writing_task'], # type: ignore[index]
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Create and return the teacher-oriented crew.

        Assembles and configures a CrewAI crew with outline generation and
        document writing agents and tasks for educational content creation.

        Returns
        -------
        Crew
            A configured CrewAI crew ready for educational content generation tasks.

        Examples
        --------
        >>> crew_instance = Teachercrew()
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
