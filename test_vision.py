import asyncio
import os
from med_safety_gym.vision_audit import get_audit_summary

async def main():
    print(f"GEMINI_API_KEY set: {bool(os.environ.get('GEMINI_API_KEY'))}")
    summary = await get_audit_summary("delete_repo", {"repo_name": "surfiniaburger/test"})
    print(f"Summary: {summary}")

if __name__ == "__main__":
    asyncio.run(main())
