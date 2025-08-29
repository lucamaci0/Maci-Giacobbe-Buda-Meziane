import os
import base64
import time
from typing import Any, Type, List
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
	prompt: str = Field(..., description="Description of the image to generate.")
	path: str = Field(..., description="Path to the folder where image should be saved.")
	security_context: Any = Field(..., description="Security context for the image generation.")

class ImageGenerationTool(BaseTool):
	"""CrewAI tool that generates images based on a description."""

	name: str = "Image Generation Tool"
	description: str = (
		"A tool to generate images based on a description."
	)
	args_schema: Type[BaseModel] = GenerateImageInput

	def _run(self, prompt: str, path: str, security_context) -> List[dict]:
		"""Run a search and return a simple formatted string of the first result."""

		load_dotenv()

		# load credentials
		client = AzureOpenAI(
				api_key=os.getenv("AZURE_API_KEY") or "",
				api_version=os.getenv("AZURE_DALLE_API_VERSION") or "",
				azure_endpoint=os.getenv("AZURE_API_BASE") or "",
		)

		# generate an image
		result = client.images.generate(
				model=os.getenv("DEPLOYMENT_IMAGE_GENERATION"),
				prompt=prompt,
				size="1024x1024",   # options: 256x256, 512x512, 1024x1024
				response_format="b64_json",
		)

		try:

			if result and result.data and len(result.data) > 0:
				image_base64 = result.data[0].b64_json
				if image_base64 is None:
					raise ValueError("No base64 image returned by the API")
				image_bytes = base64.b64decode(image_base64)

				# filename = os.path.join(path, f"generated_{int(time.time())}.png")
				filename = f"generated_{int(time.time())}.png"

				with open(filename, "wb") as f:
						f.write(image_bytes)

		except Exception as e:
			print(f"Error generating image: {e}")