#!/usr/bin/env python
"""RAG-or-Search flow entrypoint.

This module defines a CrewAI Flow that routes a user request to one of three
tools: a RAG pipeline, a web search pipeline, or a math pipeline. The flow
validates the input for safety, classifies the request, runs the appropriate
branch, and can produce an explanation.

Notes
-----
- Interactive: prompts the user for input when run as a script.
- Requires Azure OpenAI configuration via environment variables.
"""
import os
os.environ["CREWAI_TELEMETRY_DISABLED"] = "1"

from pydantic import BaseModel
from crewai import LLM

from crewai.flow import Flow, listen, start, router, or_

from src.rag_or_search.crews.searchcrew.searchcrew import SearchCrew
from src.rag_or_search.crews.ragcrew.ragcrew import Ragcrew
from src.rag_or_search.crews.mathcrew.mathcrew import Mathcrew
from src.rag_or_search.crews.teachercrew.teachercrew import Teachercrew


class RAGSearchState(BaseModel):
    """Shared state for the RAG-or-Search flow.

    Attributes
    ----------
    request : str
        The user-provided query or problem statement.
    tool : str
        The selected tool label, one of {"RAG", "web", "math"}.
    result : str
        The aggregated result string produced by the executed branch.
    """

    request: str = "" 
    tool: str = ""  # "RAG" or "web"
    result: str = ""


class RAGSearchFlow(Flow[RAGSearchState]):
    """Flow that validates, classifies, routes, and explains a user request.

    The flow performs the following steps:
    1. Collect a request and validate it for safety.
    2. Classify the request into one of RAG, web, or math.
    3. Execute the selected branch.
    4. When RAG or web is selected, explain the result using a teaching agent.
    """

    @start()
    def get_user_request(self):
        """Prompt the user, validate for safety, and classify the request.

        Returns
        -------
        str
            The validated user request text.
        """
        
        llm = LLM(model="azure/gpt-4o")
        
        while True:
            self.state.request = input("Enter your request: ")
            
            messages = [
            {
                "role": "system",
                "content": (
                "You are an AI assistant that evaluates topics for safety and ethics. "
                "Given a topic, determine if it is dangerous, unethical, or otherwise inappropriate. "
                "Respond with 'safe' if the topic is appropriate, or 'unsafe' if it is dangerous or unethical."
                )
            },
            {
                "role": "user",
                "content": f"Is the following topic safe or unsafe? Topic: '{self.state.request}'"
            }
            ]

            response = llm.call(messages=messages)
            
            if "unsafe" in response.lower():
                print("The topic is unsafe. Please enter a different topic.")
            else:
                break
            
        print("***** USER REQUEST *****")
        print(f"Request: {self.state.request}")
            
        messages = [
            {
            "role": "system",
            "content": (
                "You are an AI assistant that classifies user requests according to the following rules: "
                "1) If the user's request is related to RAG systems, output 'RAG'. "
                "2) If the user request is to compute a mathematical formula (e.g. the area of a circle, the square root of a value), output 'math'."
                "3) If the user request is about anything else (e.g., web, general topics), output 'web'."
                "Only respond with 'RAG', 'math', or 'web'."
            )
            },
            {
            "role": "user",
            "content": f"Classify the following topic: '{self.state.request}'"
            }
        ]
        
        self.state.tool = llm.call(messages=messages)
        
        print("*"*10 + self.state.tool + "*"*10)
        
        return self.state.request

    @router(get_user_request)
    def select_tool(self):
        """Return the next step label based on the classification.

        Returns
        -------
        str
            One of {"RAG", "web", "math"} which controls the next node.
        """
        
        if self.state.tool == "RAG":
            print("RAG selected to answer your query")
            return "RAG"
        elif self.state.tool == "web":
            print("Web search selected to answer your query")
            return "web"
        elif self.state.tool == "math":
            print("Math selected to answer your query")
            return "math"

    @listen("RAG")
    def query_RAG(self):
        """Execute the RAG pipeline branch.

        Returns
        -------
        CrewOutput
            The raw result from the RAG crew kickoff.
        """
        print(f"Using RAG to search for topic: '{self.state.request}'")

        self.state.result = Ragcrew().crew().kickoff(
            inputs={
                "request": self.state.request
            }
        )
        
        return self.state.result

    @listen(or_("web", query_RAG))
    def query_web(self):
        """Execute the web search branch and merge results when RAG has been previously run.

        Returns
        -------
        CrewOutput
            The result from the web search crew kickoff; may be concatenated
            with the RAG result if both were run.
        """
        result = SearchCrew().crew().kickoff(
            inputs={
                "request": self.state.request
            }
        )
        
        if self.state.tool == "RAG":
            self.state.result = result + "\n\n" + self.state.result.raw
        else:
            self.state.result = result
        
        return self.state.result
        
    @listen("math")
    def query_math(self):
        """Execute the math branch.

        Notes
        -----
        The result of this branch is produced by the math crew. If needed, you
        can store or transform it in ``self.state.result``.
        """

        _ = Mathcrew().crew().kickoff(
            inputs={
                "question": self.state.request
            }
        )
        
    @listen(query_web)
    def explain(self):
        """Run an explanatory step using a teaching agent.

        Uses the original request and the aggregated result as inputs.
        """
        
        _ = Teachercrew().crew().kickoff(
            inputs={
                "request": self.state.request,
                "info": self.state.result.raw
            }
        )
            
def kickoff():
    """Kick off the interactive RAG-or-Search flow."""
    poem_flow = RAGSearchFlow()
    poem_flow.kickoff()


def plot():
    """Render and open an interactive diagram of the flow."""
    poem_flow = RAGSearchFlow()
    poem_flow.plot()


if __name__ == "__main__":
    kickoff()
