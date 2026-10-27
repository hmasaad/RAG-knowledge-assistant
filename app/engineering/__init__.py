"""AI Engineering Task Manager: trajectories, evaluation, recovery, and replay."""

from app.engineering.classifier import FailureClass, classify_failure
from app.engineering.contracts import EVALUATOR_VERSION, EvaluationReport
from app.engineering.improve import ImprovementDecision, compare_reports
from app.engineering.recovery import RecoveryAnalysis, analyze_recovery
from app.engineering.service import EngineeringService
from app.engineering.trajectory import TrajectoryRecorder

__all__ = [
    "EVALUATOR_VERSION",
    "EngineeringService",
    "EvaluationReport",
    "FailureClass",
    "ImprovementDecision",
    "RecoveryAnalysis",
    "TrajectoryRecorder",
    "analyze_recovery",
    "classify_failure",
    "compare_reports",
]
