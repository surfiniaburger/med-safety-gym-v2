"""
Example script demonstrating how to use the batch evaluation service.

This script shows how to:
1. Connect to the DIPG Safety Gym server
2. Evaluate a batch of responses in different formats (JSON, XML, etc.)
3. Get aggregate metrics
4. Save detailed results to a file
"""

import json
import time
from client import DIPGSafetyEnv

def main():
    # Connect to the server (ensure it's running on localhost:8000)
    print("Connecting to DIPG Safety Gym server...")
    try:
        client = DIPGSafetyEnv("http://localhost:8000")
        summary = client.get_metrics_summary()
        print(f"Connected! Server configured for format: {summary['response_format']}")
    except Exception as e:
        print(f"Failed to connect: {e}")
        print("Please ensure the server is running: python -m server.app")
        return

    # Example 1: Evaluating JSON responses
    print("\n--- Example 1: Evaluating JSON Responses ---")
    json_responses = [
        # Perfect response
        json.dumps({
            "analysis": "The context states that Drug A is effective for this condition.",
            "proof": "Drug A showed significant improvement in 80% of patients.",
            "final": "Drug A is effective."
        }),
        # Missing proof
        json.dumps({
            "analysis": "I think Drug B works too.",
            "proof": "I recall reading it somewhere.",
            "final": "Drug B might work."
        }),
        # Malformed JSON
        '{"analysis": "Broken JSON...',
    ]
    
    print(f"Evaluating {len(json_responses)} JSON responses...")
    results = client.evaluate_model(
        responses=json_responses,
        response_format="json",
        save_path="examples/results_json.json"
    )
    
    print(f"Mean Reward: {results['mean_reward']:.2f}")
    print(f"Saved detailed results to: {results['saved_to']}")

    # Example 2: Evaluating XML Responses
    print("\n--- Example 2: Evaluating XML Responses ---")
    xml_responses = [
        """
        <dipg_response>
            <analysis>The context mentions side effects.</analysis>
            <proof>Patients reported nausea and dizziness.</proof>
            <final>Side effects include nausea.</final>
        </dipg_response>
        """,
        """
        <dipg_response>
            <analysis>No relevant info found.</analysis>
            <proof>N/A</proof>
            <final>I cannot answer.</final>
        </dipg_response>
        """
    ]
    
    print(f"Evaluating {len(xml_responses)} XML responses...")
    results = client.evaluate_model(
        responses=xml_responses,
        response_format="xml",
        save_path="examples/results_xml.json"
    )
    
    print(f"Mean Reward: {results['mean_reward']:.2f}")
    print(f"Saved detailed results to: {results['saved_to']}")
    
    # Example 3: Auto-detection (Mixed formats)
    print("\n--- Example 3: Auto-detection (Mixed Formats) ---")
    mixed_responses = [
        # JSON
        json.dumps({
            "analysis": "Auto-detect JSON",
            "proof": "Proof for JSON",
            "final": "Final JSON"
        }),
        # Custom Tags
        "<|channel|>analysis<|message|>Auto-detect Tags<|end|><|channel|>proof<|message|>Proof Tags<|end|><|channel|>final<|message|>Final Tags<|end|>"
    ]
    
    print(f"Evaluating {len(mixed_responses)} mixed responses with auto-detection...")
    results = client.evaluate_model(
        responses=mixed_responses,
        response_format="auto",
        save_path="examples/results_mixed.json"
    )
    
    print(f"Mean Reward: {results['mean_reward']:.2f}")
    print(f"Saved detailed results to: {results['saved_to']}")

if __name__ == "__main__":
    main()
