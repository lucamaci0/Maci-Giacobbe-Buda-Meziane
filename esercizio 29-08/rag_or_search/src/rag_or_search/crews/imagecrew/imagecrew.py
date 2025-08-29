from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from src.rag_or_search.tools.image_generation_tool import ImageGenerationTool

@CrewBase
class ImageCrew():
    """Imagecrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def scene_extractor(self) -> Agent:
        return Agent(
            config=self.agents_config['scene_extractor'], # type: ignore[index]
            verbose=True
        )

    @agent
    def image_prompt_generator(self) -> Agent:
        return Agent(
            config=self.agents_config['image_prompt_generator'], # type: ignore[index]
            verbose=True
        )

    @agent
    def image_creator(self) -> Agent:
        return Agent(
            config=self.agents_config['image_creator'], # type: ignore[index]
            verbose=True,
            tools=[ImageGenerationTool()]
        )

    @task
    def scene_extraction_task(self) -> Task:
        return Task(
            config=self.tasks_config['scene_extraction_task'], # type: ignore[index]
        )

    @task
    def image_prompt_task(self) -> Task:
        return Task(
            config=self.tasks_config['image_prompt_task']# type: ignore[index]
        )
        
    @task
    def image_creation_task(self) -> Task:
        return Task(
            config=self.tasks_config['image_creation_task'] # type: ignore[index]

        )
    @crew
    def crew(self) -> Crew:
        """Creates the Imagecrew crew"""
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
