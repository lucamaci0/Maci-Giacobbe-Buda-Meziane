import os
import base64
import time
from typing import Type, List
from pydantic import Field
from dotenv import load_dotenv
from openai import AzureOpenAI, BaseModel
from crewai.tools import BaseTool


class GenerateImageInput(BaseModel):
	"""Input schema for ``GenerateImageTool``.

	Parameters
	----------
	description : str
		Description of the image to generate.
	"""
	description: str = Field(..., description="Description of the image to generate.")

class GenerateImageTool(BaseTool):
	"""CrewAI tool that generates images based on a description."""

	name: str = "Image Generation Tool"
	description: str = (
		"A tool to generate images based on a description."
	)
	args_schema: Type[BaseModel] = GenerateImageInput

	def _run(self, description: str) -> List[dict]:
		"""Run a search and return a simple formatted string of the first result."""

		load_dotenv()

		# load credentials
		client = AzureOpenAI(
				api_key=os.getenv("AZURE_OPENAI_API_KEY") or "",
				api_version=os.getenv("AZURE_OPENAI_API_VERSION") or "",
				azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT") or "",
		)

		# generate an image
		result = client.images.generate(
				model=os.getenv("DEPLOYMENT_IMAGE_GENERATION"),
				prompt=description,
				size="1024x1024",   # options: 256x256, 512x512, 1024x1024
				response_format="b64_json",
		)

		try:

			if result and result.data and len(result.data) > 0:
				image_base64 = result.data[0].b64_json
				if image_base64 is None:
					raise ValueError("No base64 image returned by the API")
				image_bytes = base64.b64decode(image_base64)

				filename = f"generated_{int(time.time())}.png"
				with open(filename, "wb") as f:
						f.write(image_bytes)

		except Exception as e:
			print(f"Error generating image: {e}")