#!/usr/bin/env python
"""
Simple TTS Test - Standalone Main
Quick test script to verify TTS crew functionality with sample text.
"""
import os
os.environ["CREWAI_TELEMETRY_DISABLED"] = "1"

# Import your TTS crew
from crews.textspeechcrew.textspeechcrew import TTSCrew
from dotenv import load_dotenv

load_dotenv()

def main():
    """Simple test function for TTS crew"""
    
    # Sample test texts - you can modify these
    test_texts = [
        "Hello world! This is a test of our text to speech system.",
        "CrewAI is working perfectly with audio generation.",
        "The quick brown fox jumps over the lazy dog."
    ]
    
    print(" TTS Crew Test Starting...")
    print("=" * 50)
    
    # Let user choose a test text or enter their own
    print("Choose a test option:")
    for i, text in enumerate(test_texts, 1):
        print(f"{i}. {text}")
    print("4. Enter your own text")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        test_text = test_texts[0]
    elif choice == "2":
        test_text = test_texts[1]
    elif choice == "3":
        test_text = test_texts[2]
    elif choice == "4":
        test_text = input("Enter your text: ").strip()
    else:
        test_text = test_texts[0]  # Default to first option
    
    if not test_text:
        print("No text provided. Using default.")
        test_text = test_texts[0]
    
    print(f"\nConverting to speech: '{test_text}'")
    print("Processing...")
    
    try:
        # Initialize and run the TTS crew
        tts_crew = TTSCrew()
        
        result = tts_crew.crew().kickoff(
            inputs={'text': test_text}
        )
        
        print("\nTTS Test Complete!")
        print("=" * 50)
        print(" Check the 'audio_outputs' folder for your generated audio file!")
        print(f" Result: {result}")
        
    except Exception as e:
        print(f"\n Error during TTS test: {e}")
        print("    - Make sure you have:")
        print("   - Installed pyttsx3: pip install pyttsx3")
        print("   - Created the TTS crew files")
        print("   - Proper YAML configs")

if __name__ == "__main__":
    main()


# Alternative even simpler version - Direct TTS tool test
def test_tool_directly():
    """Test just the TTS tool without the full crew"""
    
    from tools.tts_tool import TTSTool
    
    print(" Testing TTS Tool Directly...")
    
    tool = TTSTool()
    test_text = "This is a direct tool test. If you hear this, the tool works!"
    
    result = tool._run(text=test_text)
    print(f"Tool Result: {result}")

# Uncomment this line to test just the tool:
#test_tool_directly()