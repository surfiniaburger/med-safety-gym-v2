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
