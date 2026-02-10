from typing import List, Optional, Any, Dict
from med_safety_eval.rubric import Rubric
from med_safety_eval.observer import RubricObserver, DataSink

def setup_rubric_observer(
    rubric: Rubric, 
    sinks: Optional[List[DataSink]], 
    session_id: Optional[str] = None,
    base_metadata: Optional[Dict[str, Any]] = None
) -> Optional[RubricObserver]:
    """
    Initializes a RubricObserver if sinks are provided.
    
    Args:
        rubric: The root rubric to observe.
        sinks: List of data sinks to emit snapshots to.
        session_id: Optional session identifier.
        base_metadata: Optional metadata to attach to all observations.
        
    Returns:
        A RubricObserver instance or None if no sinks provided.
    """
    if sinks:
        return RubricObserver(
            root_rubric=rubric,
            sinks=sinks,
            session_id=session_id or "local_eval",
            base_metadata=base_metadata
        )
    return None
