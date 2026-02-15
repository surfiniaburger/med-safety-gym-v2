import edge_tts
import logging
import os
import aiofiles

logger = logging.getLogger(__name__)

# Default voice: High quality multilingual neural
DEFAULT_VOICE = "en-US-EmmaMultilingualNeural"

async def text_to_speech(text: str, output_path: str, voice: str = DEFAULT_VOICE) -> bool:
    """
    Generate an MP3 voice note from text using Edge TTS.
    Returns: True if successful, False otherwise.
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"Successfully generated voice note: {output_path}")
            return True
        else:
            logger.error(f"Voice note generation failed (file empty or missing): {output_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error generating TTS: {e}")
        return False

async def cleanup_voice_note(path: str):
    """Delete a temporary voice note file."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logger.warning(f"Failed to cleanup voice note {path}: {e}")
