
"""
SafeClaw Intent MCP Server (Sentinel)
Categorizes user intent to inform the Mediator's context injection strategy.
Following: SafeClaw Practices (Farley & Beck)
"""

import logging
from fastmcp import FastMCP, Context
from med_safety_gym.intent_classifier import IntentClassifier, IntentCategory

# Initialize FastMCP Server
mcp = FastMCP("SafeClaw Sentinel")
logger = logging.getLogger(__name__)

# Abstraction of the intent classifier
_classifier = IntentClassifier()

@mcp.tool()
async def classify_intent(text: str, ctx: Context) -> dict:
    """
    Classifies the user's intent based on the provided text.
    Returns a dictionary with 'category' and 'is_correction'.
    """
    logger.info(f"Classifying intent for: {text[:50]}...")
    
    # Use the isolated classifier logic
    result = _classifier.classify(text)
    
    # FastMCP 3.0 allows setting state in the context if needed for the session,
    # but intent is usually per-turn.
    
    return {
        "category": result.category.name,
        "is_correction": result.is_correction
    }

@mcp.tool()
async def update_intent_rules(guidelines: str, ctx: Context) -> str:
    """
    Updates the intent classification rules with distilled guidelines.
    """
    logger.info("Updating intent classification rules with new guidelines.")
    _classifier.set_guidelines(guidelines)
    return "Intent classification rules updated."

if __name__ == "__main__":
    mcp.run()
