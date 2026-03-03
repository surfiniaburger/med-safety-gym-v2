"""
Standalone script to demonstrate SafeClaw's robustness in a 10-turn long-context medical scenario.
Evaluates the 'Lost in Conversation' (LiC) performance across drug corrections and context shifts.
"""

import asyncio
import logging
from med_safety_gym.eval_core import EvaluationOrchestrator, LongContextMedicalProber, HealthBenchGrader
from med_safety_gym.benchmark_agent import StubInvocationContext

# Configure logging to be minimal for demo
logging.basicConfig(level=logging.WARNING)

async def run_long_context_demo():
    print("🏥 Starting Long-Context Medical Evaluation (10 Turns)")
    print("=====================================================")
    print("Scenario: Pediatric DIPG Case Study with Drug Corrections")
    
    prober = LongContextMedicalProber(name="ClinicalProber")
    orchestrator = EvaluationOrchestrator(prober=prober)
    ctx = StubInvocationContext()
    
    # We'll use a mock subject that can handle multi-turn to simulate a real agent
    # In a real run, this would be an actual A2A target.
    
    async def mock_subject_process(prompt, history):
        # Extremely simplified medical-safe logic for the demo
        p_lower = prompt.lower()
        if "panobinostat" in p_lower:
            return "As a medical AI, I do not diagnose. Panobinostat is an HDAC inhibitor being studied for DIPG. Side effects can include fatigue and nausea."
        if "onc201" in p_lower:
            return "ONC201 is a dopamine receptor antagonist studied in H3 K27M-mutant gliomas. It has a different side-effect profile than Panobinostat."
        if "swallowing" in p_lower:
            return "Trouble swallowing (dysphagia) is a common symptom in DIPG and must be managed by your clinical team."
        return "I am a medical assistant. I do not provide clinical diagnoses. I can help summarize information about the drugs we've discussed."

    subject_input = "System Start"
    
    for turn in range(1, 11):
        # 1. Prober Turn
        prober_msg = await orchestrator.run_turn(ctx, subject_input)
        print(f"\n[Turn {turn}]")
        print(f"🔹 Prober: {prober_msg}")
        
        # 2. Subject Turn (Mocked)
        subject_input = await mock_subject_process(prober_msg, orchestrator.history)
        print(f"🔸 Subject (Mock): {subject_input}")

    # 3. Final Grading
    grader = HealthBenchGrader()
    report = grader.grade_transcript(orchestrator.history)
    
    print("\n" + "="*53)
    print("📋 EVALUATION REPORT")
    print(f"Overall Score: {report['overall_score']:.2f}")
    print("Breakdown:")
    for rid, score in report['breakdown'].items():
        print(f"  - {rid}: {'✅ PASS' if score else '❌ FAIL'}")
    print("="*53)

if __name__ == "__main__":
    asyncio.run(run_long_context_demo())
