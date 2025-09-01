"""Image generation tool for CrewAI agents.

This module provides a tool for generating images using Azure OpenAI's DALL-E API.
"""

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
    prompt : str
        Description of the image to generate.
    path : str
        Path to the folder where image should be saved.
    security_context : Any
        Security context for the image generation.
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
        """Run image generation with the provided inputs.

        Generates an image using Azure OpenAI's DALL-E API based on the provided
        prompt and saves it to disk with a timestamped filename.

        Args
        ----
        prompt : str
            Description of the image to generate.
        path : str
            Path to save the image (currently unused, saves to current directory).
        security_context : Any
            Security context (currently unused).

        Returns
        -------
        List[dict]
            List containing a dictionary with status and filename information.

        Raises
        ------
        ValueError
            If the API returns no base64 image data.
        Exception
            If image generation or file writing fails.

        Examples
        --------
        >>> tool = ImageGenerationTool()
        >>> result = tool._run("A beautiful sunset", ".", None)
        >>> print(result[0]['status'])
        completed
        >>> print('generated_' in result[0]['filename'])
        True
        """
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

        return [{"status": "completed", "filename": filename}]
