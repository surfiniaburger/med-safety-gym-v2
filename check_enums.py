
try:
    from a2a.types import TaskState
    print([m.name for m in TaskState])
except ImportError:
    print("Could not import a2a.types")
