from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import pyttsx3
import os
import tempfile
from pathlib import Path
from datetime import datetime

class TTSToolSchema(BaseModel):
    """Input for TTS Tool."""
    text: str = Field(..., description="The text to convert to speech")
    output_filename: str = Field(default="", description="Name for the output audio file (optional)")
    voice_rate: int = Field(default=200, description="Speech rate (words per minute)")

class TTSTool(BaseTool):
    name: str = "Text to Speech Converter"
    description: str = "Converts text to speech and saves as an audio file with timestamp"
    args_schema: Type[BaseModel] = TTSToolSchema

    def _run(self, text: str, output_filename: str = "", voice_rate: int = 200) -> str:
        try:
            # Initialize the TTS engine
            engine = pyttsx3.init()
            
            # Configure voice properties
            engine.setProperty('rate', voice_rate)
            voices = engine.getProperty('voices')
            if voices:
                # Use the first available voice (you can modify this to select specific voices)
                engine.setProperty('voice', voices[0].id)
            
            # Create output directory if it doesn't exist
            output_dir = Path("audio_outputs")
            output_dir.mkdir(exist_ok=True)
            
            # Generate filename with timestamp if not provided
            if not output_filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"tts_output_{timestamp}.wav"
            
            # Ensure proper file extension
            if not output_filename.endswith(('.wav', '.mp3')):
                output_filename += '.wav'
            
            # Full path for the output file
            output_path = output_dir / output_filename
            
            # Convert text to speech and save
            engine.save_to_file(text, str(output_path))
            engine.runAndWait()
            
            # Verify file was created
            if output_path.exists():
                file_size = output_path.stat().st_size
                return f"Successfully converted text to speech!\\n File: {output_path}\\n Size: {file_size} bytes\\n Text: '{text[:100]}...'"
            else:
                return f" Audio file was not created. Check TTS engine configuration."
            
        except Exception as e:
            return f"Error converting text to speech: {str(e)}"
