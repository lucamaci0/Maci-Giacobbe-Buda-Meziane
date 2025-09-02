"""Crew definition for a RAG-oriented workflow."""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
# from src.rag_or_search.tools.rag import RagTool
from src.rag_or_search.tools.RAG_qdrant_new.rag_qdrant_tool import RagTool
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class Ragcrew():
    """Crew that retrieves contexts via RAG and composes a response."""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def rag_searcher(self) -> Agent:
        """Agent that performs retrieval using RagTool.

        Creates a RAG search agent configured with the RagTool to retrieve
        relevant document contexts from a vector store.

        Returns
        -------
        Agent
            A configured CrewAI agent with RAG search capabilities.

        Examples
        --------
        >>> crew = Ragcrew()
        >>> agent = crew.rag_searcher()
        >>> print(type(agent))
        <class 'crewai.agent.Agent'>
        """
        return Agent(
            config=self.agents_config['rag_searcher'], # type: ignore[index]
            verbose=True,
            tools=[RagTool()]  # Adding the RAG tool to the agent
        )

    @agent
    def rag_responder(self) -> Agent:
        """Agent that drafts a grounded response based on retrieved contexts.

        Creates a RAG response agent configured to generate answers based on
        retrieved document contexts, ensuring responses are grounded in the source material.

        Returns
        -------
        Agent
            A configured CrewAI agent specialized in generating grounded responses.

        Examples
        --------
        >>> crew = Ragcrew()
        >>> agent = crew.rag_responder()
        >>> print(type(agent))
        <class 'crewai.agent.Agent'>
        """
        return Agent(
            config=self.agents_config['rag_responder'], # type: ignore[index]
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def rag_search_task(self) -> Task:
        """Task that retrieves relevant contexts from the corpus.

        Creates a RAG search task configured to retrieve relevant document
        contexts from the vector store for a given query.

        Returns
        -------
        Task
            A configured CrewAI task for RAG search operations.

        Examples
        --------
        >>> crew = Ragcrew()
        >>> task = crew.rag_search_task()
        >>> print(type(task))
        <class 'crewai.task.Task'>
        """
        return Task(
            config=self.tasks_config['rag_search_task'], # type: ignore[index]
        )

    @task
    def rag_response_task(self) -> Task:
        """Task that synthesizes an answer grounded in retrieved contexts.

        Creates a RAG response task configured to synthesize answers based on
        retrieved contexts and save the output to a report file.

        Returns
        -------
        Task
            A configured CrewAI task for RAG response generation operations.

        Examples
        --------
        >>> crew = Ragcrew()
        >>> task = crew.rag_response_task()
        >>> print(type(task))
        <class 'crewai.task.Task'>
        """
        return Task(
            config=self.tasks_config['rag_response_task'], # type: ignore[index]
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Create and return the RAG-oriented crew.

        Assembles and configures a CrewAI crew with RAG search and response
        agents and tasks for retrieval-augmented generation workflows.

        Returns
        -------
        Crew
            A configured CrewAI crew ready for RAG operations.

        Examples
        --------
        >>> crew_instance = Ragcrew()
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
