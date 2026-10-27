"""AI Engineering Task Manager: trajectories, evaluation, recovery, and replay."""

from app.engineering.classifier import FailureClass, classify_failure
from app.engineering.contracts import EVALUATOR_VERSION, EvaluationReport
from app.engineering.improve import ImprovementDecision, compare_reports
from app.engineering.governance import ProductionGovernance
from app.engineering.loop import ContinuousImprovementLoop, compare_metrics
from app.engineering.recovery import RecoveryAnalysis, analyze_recovery
from app.engineering.service import EngineeringService
from app.engineering.trajectory import TrajectoryRecorder

__all__ = [
    "EVALUATOR_VERSION",
    "ContinuousImprovementLoop",
    "EngineeringService",
    "EvaluationReport",
    "FailureClass",
    "ImprovementDecision",
    "ProductionGovernance",
    "RecoveryAnalysis",
    "TrajectoryRecorder",
    "analyze_recovery",
    "classify_failure",
    "compare_metrics",
    "compare_reports",
]
