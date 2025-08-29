"""CrewAI tool that wraps simple RAG retrieval utilities.

Accepts a question and a ``k`` value to retrieve top-k contexts from a local
vector store. Useful as an agent tool step before generation.
"""

from typing import Type, List
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from .rag_utils import rag_search

class RagToolInput(BaseModel):
	"""Input schema for ``RagTool``.

	Parameters
	----------
	question : str
		Question to answer using RAG search.
	k : int
		Number of documents to retrieve for context.
	"""
	question: str = Field(..., description="Question to answer using RAG search.")
	k: int = Field(3, description="Number of documents to retrieve for context.")
 
class RagTool(BaseTool):
	"""CrewAI tool that performs a simple RAG retrieval.

	Notes
	-----
	The tool returns contexts as a mapping of ``source`` to text and does not
	perform generation.
	"""

	name: str = "RAG Search Tool"
	description: str = (
		"A tool that performs a Retrieval-Augmented Generation (RAG) search given a question and a number of documents to retrieve. "
		"Uses a local vector store and LLM to retrieve and answer based on context."
		"Returns a dictionary in the form { 'source': str, 'document': str }"
	)
	args_schema: Type[BaseModel] = RagToolInput

	def _run(self, question: str, k: int) -> List[str]:
		"""Run retrieval with the provided inputs.

		Parameters
		----------
		question : str
			The query to retrieve contexts for.
		k : int
			Number of contexts to retrieve.

		Returns
		-------
		dict
			Mapping of ``source`` to ``page_content``.

		Raises
		------
		ValueError
			If ``question`` is empty.
		"""
		if not question:
			raise ValueError("Please provide a question for RAG search.")
		results = rag_search(question, k=k)
		
		return results
