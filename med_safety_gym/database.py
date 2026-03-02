from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from datetime import datetime
import os

Base = declarative_base()

class TrainingSession(Base):
    __tablename__ = 'training_sessions'
    
    id = Column(String, primary_key=True) # User-provided or UUID
    name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    steps = relationship("TrajectoryStep", back_populates="session")

class ConversationSession(Base):
    """Persists SafeClaw conversation memory and settings."""
    __tablename__ = 'conversation_sessions'
    
    id = Column(String, primary_key=True) # chat_id
    github_repo = Column(String, nullable=True)
    messages_json = Column(Text, default="[]") # JSON serialized history
    pending_action_json = Column(Text, nullable=True) # HITL state
    escalated_tools_json = Column(Text, default="[]") # Persisted escalation
    last_active = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ContrastivePair(Base):
    """Stores successful (D+) or failed (D-) interaction trajectories for the Experience Refiner."""
    __tablename__ = 'contrastive_pairs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String) # For linking to a user's habits
    trajectory_json = Column(Text) # The conversation turns leading to the outcome
    is_success = Column(Integer) # 1 for D+ (success), 0 for D- (failure)
    created_at = Column(DateTime, default=datetime.utcnow)

class TrajectoryStep(Base):
    __tablename__ = 'trajectory_steps'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey('training_sessions.id'))
    step_index = Column(Integer)
    action = Column(Text)
    observation = Column(Text)
    total_reward = Column(Float)
    
    session = relationship("TrainingSession", back_populates="steps")
    rubric_scores = relationship("RubricScore", back_populates="step")

class RubricScore(Base):
    __tablename__ = 'rubric_scores'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    step_id = Column(Integer, ForeignKey('trajectory_steps.id'))
    path = Column(String) # e.g. "grounding.fuzzy_match"
    score = Column(Float)
    
    step = relationship("TrajectoryStep", back_populates="rubric_scores")

# Database Connection
# Default to sqlite for local dev if POSTGRES_URL not set
DATABASE_URL = os.environ.get("POSTGRES_URL", "sqlite:///./med_safety_gym.db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
