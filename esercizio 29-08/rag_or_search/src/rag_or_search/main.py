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

from pydantic import BaseModel
from crewai import LLM
from crewai.flow import Flow, listen, start, router, or_

from src.rag_or_search.crews.searchcrew.searchcrew import SearchCrew
from src.rag_or_search.crews.ragcrew.ragcrew import Ragcrew
from src.rag_or_search.crews.mathcrew.mathcrew import Mathcrew
from src.rag_or_search.crews.teachercrew.teachercrew import Teachercrew
from src.rag_or_search.crews.imagecrew.imagecrew import ImageCrew
from src.rag_or_search.crews.aiactcrew.aiactcrew import Aiactcrew

os.environ["CREWAI_TELEMETRY_DISABLED"] = "1"

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
    input: int = 0


class RAGSearchFlow(Flow[RAGSearchState]):
    """Flow that validates, classifies, routes, and explains a user request.

    The flow performs the following steps:
    1. Collect a request and validate it for safety.
    2. Classify the request into one of RAG, web, or math.
    3. Execute the selected branch.
    4. When RAG or web is selected, explain the result using a teaching agent.
    """

    @start()
    def select_option(self):

        valid_option = False
        while not valid_option:
            self.state.input = int(input("Select an option: 1. Generate AI act report, 2. Run RAG-or-Search flow"))
            if self.state.input == 1:
                valid_option = True
            elif self.state.input == 2:
                valid_option = True
            else:
                print("Invalid option")
        return self.state.input
                
    @router(select_option)
    def route(self):
        
        if self.state.input == 1:
            return "ai_act"
        elif self.state.input == 2:
            return "rag_or_search"

    @listen("rag_or_search")
    def get_user_request(self):
        """Prompt the user, validate for safety, and classify the request.

        This method handles the initial user interaction by prompting for input,
        validating the request for safety using an LLM, and classifying the
        request into one of three categories: RAG, web, or math.

        Returns
        -------
        str
            The validated user request text.

        Raises
        ------
        SystemExit
            If the user provides an unsafe topic and doesn't provide an alternative.

        Examples
        --------
        >>> flow = RAGSearchFlow()
        >>> # User enters: "What is machine learning?"
        >>> # Method validates and classifies as "web"
        >>> result = flow.get_user_request()
        >>> print(flow.state.tool)
        web
        """

        llm = LLM(model="azure/gpt-4o")

        while True:
            self.state.request = input("Enter your request: ")

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant that evaluates topics for safety and ethics. "
                        "Given a topic, determine if it is dangerous, unethical, or otherwise "
                        "inappropriate. Respond with 'safe' if the topic is appropriate, or "
                        "'unsafe' if it is dangerous or unethical."
                    )
                },
                {
                    "role": "user",
                    "content": f"Is the following topic safe or unsafe? "
                            f"Topic: '{self.state.request}'"
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
                    "You are an AI assistant that classifies user requests according to "
                    "the following rules: 1) If the user's request is related to RAG "
                    "systems, output 'RAG'. 2) If the user request is to compute a "
                    "mathematical formula (e.g. the area of a circle, the square root "
                    "of a value), output 'math'. 3) If the user request is about "
                    "anything else (e.g., web, general topics), output 'web'. "
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
    
    @listen("ai_act")
    def generate_ai_act(self):
        """Generate an AI Act compliance report for the flow."""
        print("SONO QUI DENTRO GENERATE!!!")
        report = Aiactcrew().crew().kickoff()
        print(report)
        
        return report

    @router(get_user_request)
    def select_tool(self):
        """Return the next step label based on the classification.

        Routes the flow to the appropriate branch based on the tool classification
        performed in the previous step. Prints a confirmation message and returns
        the tool name for flow routing.

        Returns
        -------
        str or None
            One of {"RAG", "web", "math"} which controls the next node, or None
            if no valid tool is selected.

        Examples
        --------
        >>> flow = RAGSearchFlow()
        >>> flow.state.tool = "RAG"
        >>> result = flow.select_tool()
        >>> print(result)
        RAG
        """
        if self.state.tool == "RAG":
            print("RAG selected to answer your query")
            return "RAG"
        if self.state.tool == "web":
            print("Web search selected to answer your query")
            return "web"
        if self.state.tool == "math":
            print("Math selected to answer your query")
            return "math"
        return None


    @listen("RAG")
    def query_rag(self):
        """Execute the RAG pipeline branch.

        Runs the RAG crew to retrieve relevant contexts from the vector store
        and generate a response based on the retrieved information.

        Returns
        -------
        CrewOutput
            The raw result from the RAG crew kickoff containing the generated response.

        Examples
        --------
        >>> flow = RAGSearchFlow()
        >>> flow.state.request = "What is LangChain?"
        >>> result = flow.query_rag()
        >>> print(type(result))
        <class 'crewai.crew.CrewOutput'>
        """
        print(f"Using RAG to search for topic: '{self.state.request}'")

        self.state.result = Ragcrew().crew().kickoff(
            inputs={
                "request": self.state.request
            }
        )

        return self.state.result

    @listen(or_("web", query_rag))
    def query_web(self):
        """Execute the web search branch and merge results when RAG has been previously run.

        Performs web search using the SearchCrew and optionally merges results
        with previous RAG results if both branches have been executed.

        Returns
        -------
        CrewOutput
            The result from the web search crew kickoff; may be concatenated
            with the RAG result if both were run.

        Examples
        --------
        >>> flow = RAGSearchFlow()
        >>> flow.state.request = "Latest AI news"
        >>> result = flow.query_web()
        >>> print(type(result))
        <class 'crewai.crew.CrewOutput'>
        """
        result = SearchCrew().crew().kickoff(
            inputs={
                "request": self.state.request
            }
        )

        if self.state.tool == "RAG":
            self.state.result = result.raw + "\n\n" + self.state.result.raw
        else:
            self.state.result = result

        return self.state.result

    @listen("math")
    def query_math(self):
        """Execute the math branch.

        Runs the MathCrew to translate mathematical problems into code and
        execute them using a code interpreter tool.

        Returns
        -------
        CrewOutput
            The result from the math crew kickoff containing the calculated result.

        Examples
        --------
        >>> flow = RAGSearchFlow()
        >>> flow.state.request = "Calculate the area of a circle with radius 5"
        >>> result = flow.query_math()
        >>> print(type(result))
        <class 'crewai.crew.CrewOutput'>
        """
        _ = Mathcrew().crew().kickoff(
            inputs={
                "question": self.state.request
            }
        )

    @listen(query_web)
    def explain(self):
        """Run an explanatory step using a teaching agent.

        Uses the TeacherCrew to generate an explanatory document based on the
        original request and the aggregated results from previous steps.

        Returns
        -------
        CrewOutput
            The result from the teacher crew kickoff containing the explanation.

        Examples
        --------
        >>> flow = RAGSearchFlow()
        >>> flow.state.request = "Explain machine learning"
        >>> flow.state.result = mock_result
        >>> result = flow.explain()
        >>> print(type(result))
        <class 'crewai.crew.CrewOutput'>
        """
        _ = Teachercrew().crew().kickoff(
            inputs={
                "request": self.state.request,
                "info": self.state.result.raw
            }
        )

    @listen(explain)
    def generate_image(self):
        """Generate an image based on the user request.

        Uses the ImageCrew to create a visual representation of the topic
        discussed in the previous steps.

        Returns
        -------
        CrewOutput
            The result from the image crew kickoff containing the generated image.

        Examples
        --------
        >>> flow = RAGSearchFlow()
        >>> flow.state.request = "Create an image of a neural network"
        >>> flow.state.result = mock_result
        >>> result = flow.generate_image()
        >>> print(type(result))
        <class 'crewai.crew.CrewOutput'>
        """
        _ = ImageCrew().crew().kickoff(
            inputs={
                "topic": self.state.request,
                "text": self.state.result.raw
            }
        )
def kickoff():
    """Kick off the interactive RAG-or-Search flow.

    Creates a new RAGSearchFlow instance and starts the interactive flow,
    prompting the user for input and processing their request through the
    appropriate pipeline.

    Examples
    --------
    >>> kickoff()
    Enter your request: What is artificial intelligence?
    ***** USER REQUEST *****
    Request: What is artificial intelligence?
    **********web**********
    Web search selected to answer your query
    """
    poem_flow = RAGSearchFlow()
    poem_flow.kickoff()


def plot():
    """Render and open an interactive diagram of the flow.

    Creates a RAGSearchFlow instance and generates a visual representation
    of the flow structure, opening it in the default web browser.

    Examples
    --------
    >>> plot()
    # Opens a browser window with the flow diagram
    """
    poem_flow = RAGSearchFlow()
    poem_flow.plot()


if __name__ == "__main__":
    kickoff()
    # generate_ai_act()
