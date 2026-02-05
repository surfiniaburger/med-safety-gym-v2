from typing import List, Optional, Any
from med_safety_eval.rubric import Rubric
from med_safety_eval.observer import RubricObserver, DataSink

def setup_rubric_observer(
    rubric: Rubric, 
    sinks: Optional[List[DataSink]], 
    session_id: Optional[str] = None
) -> Optional[RubricObserver]:
    """
    Initializes a RubricObserver if sinks are provided.
    
    Args:
        rubric: The root rubric to observe.
        sinks: List of data sinks to emit snapshots to.
        session_id: Optional session identifier.
        
    Returns:
        A RubricObserver instance or None if no sinks provided.
    """
    if sinks:
        return RubricObserver(
            root_rubric=rubric,
            sinks=sinks,
            session_id=session_id or "local_eval"
        )
    return None
