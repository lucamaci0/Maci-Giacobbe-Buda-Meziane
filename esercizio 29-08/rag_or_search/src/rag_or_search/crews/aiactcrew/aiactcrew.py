from crewai.project import CrewBase, agent, crew, task
from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
# TODO: Import tools when they are implemented
from src.rag_or_search.tools.doc_loader import DocLoaderTool
from src.rag_or_search.tools.template_loader import TemplateLoaderTool
from src.rag_or_search.tools.RAG_qdrant_new.rag_qdrant_tool import RagTool

from crewai_tools import FirecrawlScrapeWebsiteTool
# from tools.act_generator import ActGeneratorTool

@CrewBase
class Aiactcrew():
    """Aiactcrew crew for AI Act compliance document generation"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended

    @agent
    def doc_parser(self) -> Agent:
        return Agent(
            config=self.agents_config['doc_parser'], # type: ignore[index]
            verbose=True,
            tools=[DocLoaderTool()]  # TODO: Add when tool is implemented
        )

    @agent
    def template_parser(self) -> Agent:
        return Agent(
            config=self.agents_config['template_parser'], # type: ignore[index]
            verbose=True,
            # tools=[FirecrawlScrapeWebsiteTool(url='{url}')]  # TODO: Add when tool is implemented
            tools=[DocLoaderTool()]  # TODO: Add when tool is implemented
        )

    @agent
    def act_document_generator(self) -> Agent:
        return Agent(
            config=self.agents_config['act_document_generator'], # type: ignore[index]
            verbose=True,
            # tools=[ActGeneratorTool()]  # TODO: Add when tool is implemented
        )
        
    @agent
    def rag_placeholder_filler(self) -> Agent:
        return Agent(
            config=self.agents_config['rag_placeholder_filler'], # type: ignore[index]
            verbose=True,
            tools=[RagTool()]  # TODO: Add when tool is implemented
        )
        
    @task
    def parse_docs(self) -> Task:
        return Task(
            config=self.tasks_config['parse_docs'], # type: ignore[index]
            # agent=self.doc_parser()
        )

    @task
    def parse_template(self) -> Task:
        return Task(
            config=self.tasks_config['parse_template'], # type: ignore[index]
            # agent=self.template_parser()
        )

    @task
    def generate_ai_act(self) -> Task:
        return Task(
            config=self.tasks_config['generate_ai_act'], # type: ignore[index]
            # agent=self.act_document_generator(), 
            output_file='outputs/ai_act_with_placeholders.md'
        )
    
    @task    
    def fill_placeholders(self) -> Task:
        return Task(
            config=self.tasks_config['fill_placeholders'], # type: ignore[index]
            # agent=self.act_document_generator(), 
            output_file='outputs/ai_act.md'
        ) 

    @crew
    def crew(self) -> Crew:
        """Creates the Aiactcrew crew for AI Act compliance document generation"""
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )