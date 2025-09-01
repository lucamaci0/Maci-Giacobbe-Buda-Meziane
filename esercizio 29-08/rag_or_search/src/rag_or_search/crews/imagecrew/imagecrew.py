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
        """Agent that extracts key scenes from text content.

        Creates a scene extraction agent configured to identify and extract
        important visual scenes from textual content for image generation.

        Returns
        -------
        Agent
            A configured CrewAI agent specialized in scene extraction.

        Examples
        --------
        >>> crew = ImageCrew()
        >>> agent = crew.scene_extractor()
        >>> print(type(agent))
        <class 'crewai.agent.Agent'>
        """
        return Agent(
            config=self.agents_config['scene_extractor'], # type: ignore[index]
            verbose=True
        )

    @agent
    def image_prompt_generator(self) -> Agent:
        """Agent that generates detailed image prompts from scenes.

        Creates an image prompt generation agent configured to convert
        extracted scenes into detailed prompts suitable for image generation.

        Returns
        -------
        Agent
            A configured CrewAI agent specialized in prompt generation.

        Examples
        --------
        >>> crew = ImageCrew()
        >>> agent = crew.image_prompt_generator()
        >>> print(type(agent))
        <class 'crewai.agent.Agent'>
        """
        return Agent(
            config=self.agents_config['image_prompt_generator'], # type: ignore[index]
            verbose=True
        )

    @agent
    def image_creator(self) -> Agent:
        """Agent that creates images using the image generation tool.

        Creates an image creation agent configured with the ImageGenerationTool
        to generate images based on detailed prompts.

        Returns
        -------
        Agent
            A configured CrewAI agent with image generation capabilities.

        Examples
        --------
        >>> crew = ImageCrew()
        >>> agent = crew.image_creator()
        >>> print(type(agent))
        <class 'crewai.agent.Agent'>
        """
        return Agent(
            config=self.agents_config['image_creator'], # type: ignore[index]
            verbose=True,
            tools=[ImageGenerationTool()]
        )

    @task
    def scene_extraction_task(self) -> Task:
        """Task that extracts visual scenes from text content.

        Creates a scene extraction task configured to identify and extract
        key visual elements from textual content for image generation.

        Returns
        -------
        Task
            A configured CrewAI task for scene extraction operations.

        Examples
        --------
        >>> crew = ImageCrew()
        >>> task = crew.scene_extraction_task()
        >>> print(type(task))
        <class 'crewai.task.Task'>
        """
        return Task(
            config=self.tasks_config['scene_extraction_task'], # type: ignore[index]
        )

    @task
    def image_prompt_task(self) -> Task:
        """Task that generates detailed image prompts from scenes.

        Creates an image prompt generation task configured to convert
        extracted scenes into detailed prompts for image generation.

        Returns
        -------
        Task
            A configured CrewAI task for prompt generation operations.

        Examples
        --------
        >>> crew = ImageCrew()
        >>> task = crew.image_prompt_task()
        >>> print(type(task))
        <class 'crewai.task.Task'>
        """
        return Task(
            config=self.tasks_config['image_prompt_task']# type: ignore[index]
        )
        
    @task
    def image_creation_task(self) -> Task:
        """Task that creates images using generated prompts.

        Creates an image creation task configured to generate images
        using the ImageGenerationTool with detailed prompts.

        Returns
        -------
        Task
            A configured CrewAI task for image creation operations.

        Examples
        --------
        >>> crew = ImageCrew()
        >>> task = crew.image_creation_task()
        >>> print(type(task))
        <class 'crewai.task.Task'>
        """
        return Task(
            config=self.tasks_config['image_creation_task'] # type: ignore[index]

        )
    @crew
    def crew(self) -> Crew:
        """Creates the ImageCrew crew.

        Assembles and configures a CrewAI crew with scene extraction, prompt
        generation, and image creation agents and tasks for automated image generation.

        Returns
        -------
        Crew
            A configured CrewAI crew ready for image generation tasks.

        Examples
        --------
        >>> crew_instance = ImageCrew()
        >>> crew = crew_instance.crew()
        >>> print(type(crew))
        <class 'crewai.crew.Crew'>
        """
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
