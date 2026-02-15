"""
Demonstration: SafeClaw Session Memory in Action
Shows how conversation context enables dynamic Entity Parity checks.
"""
import asyncio
from med_safety_gym.session_memory import SessionMemory


async def demo_session_memory():
    """Demonstrate session memory and context building."""
    
    print("=" * 60)
    print("SafeClaw Session Memory Demo")
    print("=" * 60)
    
    # Create a session for a Telegram user
    session = SessionMemory(user_id="telegram_12345")
    print("\n1️⃣  Created session for user: telegram_12345")
    print(f"   Initial messages: {len(session.get_messages())}")
    
    # Simulate a conversation
    print("\n2️⃣  Simulating patient conversation...")
    session.add_message("user", "Patient is 7 years old, diagnosed with DIPG")
    session.add_message("assistant", "I understand. DIPG is challenging...")
    session.add_message("user", "What treatments are available?")
    session.add_message("assistant", "Options include Panobinostat and ONC201")
    
    print(f"  Messages in history: {len(session.get_messages())}")
    
    # Extract medical context
    print("\n3️⃣  Extracting medical context from conversation...")
    context = session.get_medical_context()
    print(f"   {context}")
    
    # Show known entities
    entities = session.get_known_entities()
    print(f"\n4️⃣  Extracted {len(entities)} medical entities:")
    for entity in sorted(entities):
        print(f"   - {entity}")
    
    # Demonstrate dynamic allowlist
    print("\n5️⃣  Dynamic Entity Parity Check:")
    test_actions = [
        "Prescribe Panobinostat",
        "Enroll in ONC201 trial",
        "Prescribe ScillyCure",  # Should fail
    ]
    
    for action in test_actions:
        # Simple check: does action contain known entities?
        action_lower = action.lower()
        has_known = any(entity in action_lower for entity in entities)
        status = "✅ PASS" if has_known else "🚫 BLOCK"
        print(f"   {status}: {action}")
    
    print("\n" + "=" * 60)
    print("✨ Session memory enables context-aware safety checks!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo_session_memory())
