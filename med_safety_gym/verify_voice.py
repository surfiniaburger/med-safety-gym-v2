import asyncio
from med_safety_gym.voice import text_to_speech
import os

async def test_voice():
    test_file = "/tmp/voice_test.mp3"
    print(f"Generating voice note to {test_file}...")
    success = await text_to_speech("Testing SafeClaw Voice Integration. This is an automated safety alert.", test_file)
    if success and os.path.exists(test_file):
        print(f"✅ Success! Voice note generated. Size: {os.path.getsize(test_file)} bytes")
        # cleanup
        os.remove(test_file)
    else:
        print("❌ Failed to generate voice note.")

if __name__ == "__main__":
    asyncio.run(test_voice())
