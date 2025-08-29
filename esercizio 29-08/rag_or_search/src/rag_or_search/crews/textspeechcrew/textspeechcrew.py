from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from tools.tts_tool import TTSTool
from typing import List

@CrewBase
class TTSCrew():
    '''CrewAI project for converting text input into high-quality audio output'''

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    tts_tool = TTSTool()

    @agent
    def text_cleaner(self) -> Agent:
        return Agent(
            config=self.agents_config['text_cleaner'],
            verbose=True,
        )

    @agent
    def tts_converter(self) -> Agent:
        return Agent(
            config=self.agents_config['tts_converter'],
            tools=[TTSTool()], 
            verbose=True,
        )

    @agent
    def quality_checker(self) -> Agent:
        return Agent(
            config=self.agents_config['quality_checker'],
            verbose=True,
        )

    @task
    def clean_text_task(self) -> Task:
        return Task(
            config=self.tasks_config['clean_text_task'],
            agent=self.text_cleaner()
        )

    @task
    def tts_task(self) -> Task:
        return Task(
            config=self.tasks_config['tts_task'],
            agent=self.tts_converter()
        )

    @task
    def quality_check_task(self) -> Task:
        return Task(
            config=self.tasks_config['quality_check_task'],
            agent=self.quality_checker()
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )