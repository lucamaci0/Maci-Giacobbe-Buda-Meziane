"""Crew definition for a web search and summarization workflow."""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from src.rag_or_search.tools.search import SearchTool
from crewai_tools import SerperDevTool

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class SearchCrew():
    """Crew that searches the web and summarizes the results."""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def web_search_agent(self) -> Agent:
        """Agent that queries DuckDuckGo for relevant pages.

        Creates a web search agent configured to use SerperDevTool for
        performing web searches and retrieving relevant information.

        Returns
        -------
        Agent
            A configured CrewAI agent with web search capabilities.

        Examples
        --------
        >>> crew = SearchCrew()
        >>> agent = crew.web_search_agent()
        >>> print(type(agent))
        <class 'crewai.agent.Agent'>
        """
        search_tool = SerperDevTool()
        
        return Agent(
            config=self.agents_config['web_search_agent'], # type: ignore[index]
            verbose=True,
            tools=[search_tool]
        )

    @agent
    def summarizer(self) -> Agent:
        """Agent that condenses search results into a concise summary.

        Creates a summarization agent configured to process and condense
        web search results into clear, concise summaries.

        Returns
        -------
        Agent
            A configured CrewAI agent specialized in summarization tasks.

        Examples
        --------
        >>> crew = SearchCrew()
        >>> agent = crew.summarizer()
        >>> print(type(agent))
        <class 'crewai.agent.Agent'>
        """
        return Agent(
            config=self.agents_config['summarizer'], # type: ignore[index]
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def web_search(self) -> Task:
        """Task that gathers the top search results for a topic.

        Creates a web search task configured to gather the most relevant
        search results for a given topic using the web search agent.

        Returns
        -------
        Task
            A configured CrewAI task for web search operations.

        Examples
        --------
        >>> crew = SearchCrew()
        >>> task = crew.web_search()
        >>> print(type(task))
        <class 'crewai.task.Task'>
        """
        return Task(
            config=self.tasks_config['web_search'], # type: ignore[index]
        )

    @task
    def summarize_results(self) -> Task:
        """Task that writes a brief summary of the search findings.

        Creates a summarization task configured to process search results
        and generate a concise summary, saving the output to a report file.

        Returns
        -------
        Task
            A configured CrewAI task for summarizing search results.

        Examples
        --------
        >>> crew = SearchCrew()
        >>> task = crew.summarize_results()
        >>> print(type(task))
        <class 'crewai.task.Task'>
        """
        return Task(
            config=self.tasks_config['summarize_results'], # type: ignore[index]
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Create and return the web-search crew.

        Assembles and configures a CrewAI crew with web search and summarization
        agents and tasks, using sequential processing for coordinated execution.

        Returns
        -------
        Crew
            A configured CrewAI crew ready for web search and summarization tasks.

        Examples
        --------
        >>> crew_instance = SearchCrew()
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
