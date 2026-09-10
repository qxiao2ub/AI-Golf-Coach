"""Rule-grounded coaching suggestions, practice planning, and feedback learning."""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Tuple

import pandas as pd

from .geometry import clamp

ADVICE_LIBRARY: Dict[str, Dict[str, str]] = {
    "Posture": {
        "insight": "The address posture or knee-flex indicator is outside the prototype reference band.",
        "tip": "Build a repeatable address with softly flexed knees, a long spine, and pressure centered over the mid-foot.",
        "drill": "Mirror setup holds: hold address for 10 seconds, reset, and repeat eight times.",
    },
    "Balance": {
        "insight": "The head or hip center moved more than the prototype's preferred range.",
        "tip": "Keep pressure moving inside the feet and finish in a position you can hold without stepping.",
        "drill": "Feet-together half swings followed by a three-second balanced finish.",
    },
    "Rotation": {
        "insight": "The shoulder-to-hip separation signal suggests limited or excessive relative turn.",
        "tip": "Let the chest and pelvis turn in sequence rather than forcing isolated upper-body rotation.",
        "drill": "Club-across-chest turns: rehearse ten slow turns while keeping the knees athletic.",
    },
    "Tempo": {
        "insight": "The measured backswing-to-downswing timing is outside the broad prototype band.",
        "tip": "Use an unhurried backswing and allow speed to build through the downswing.",
        "drill": "Three-to-one count drill: count 1-2-3 to the top and 1 through impact.",
    },
    "Arm extension": {
        "insight": "The average elbow angle near the detected impact frame suggests a collapsed or over-rigid arm pattern.",
        "tip": "Create width without locking the elbows and allow the arms to extend naturally after contact.",
        "drill": "Lead-arm half swings with a soft grip and waist-high finish.",
    },
    "Consistency": {
        "insight": "The hand-speed trace changed abruptly, which can reflect sequencing inconsistency or tracking noise.",
        "tip": "Rehearse at a speed where the sequence stays stable before adding effort.",
        "drill": "Three-speed ladder: five swings each at 40%, 60%, and 75% effort.",
    },
    "Pose confidence": {
        "insight": "Landmark visibility or frame coverage was limited, so angle estimates are less reliable.",
        "tip": "Place the camera on a tripod, keep the whole body visible, and avoid strong backlighting.",
        "drill": "Recording reset: capture a five-second test clip and verify all joints remain in frame.",
    },
}

RL_ACTIONS: Dict[str, List[str]] = {
    "Posture": ["Mirror setup holds", "Alignment-stick setup checks", "Slow-motion address resets"],
    "Balance": ["Feet-together half swings", "Hold-the-finish drill", "Step-through transition drill"],
    "Rotation": ["Club-across-chest turns", "Split-stance rotation drill", "Half-swing sequencing drill"],
    "Tempo": ["Three-to-one count drill", "Metronome rehearsal", "Pause-at-the-top drill"],
    "Arm extension": ["Lead-arm half swings", "Towel-target extension drill", "Trail-hand-only soft swings"],
    "Consistency": ["Three-speed ladder", "Nine identical rehearsals", "Start-line gate drill"],
    "Pose confidence": ["Tripod recording reset", "Full-body framing check", "Lighting and clothing check"],
}


def build_advice(component_scores: Dict[str, float], summary: Dict[str, float]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    ordered = sorted(component_scores.items(), key=lambda item: item[1])
    for priority, (area, score) in enumerate(ordered[:4], start=1):
        content = ADVICE_LIBRARY[area]
        rows.append({
            "Priority": priority,
            "Area": area,
            "Component score": round(float(score), 1),
            "Insight": content["insight"],
            "Suggestion": content["tip"],
            "Practice drill": content["drill"],
        })
    return pd.DataFrame(rows)


def build_practice_plan(advice: pd.DataFrame, profile: Dict[str, Any]) -> pd.DataFrame:
    gym_days = int(profile.get("gym_activity_days_per_week", 0) or 0)
    experience = str(profile.get("experience_level", "Beginner"))
    session_minutes = 25 if experience == "Beginner" else 35 if experience == "Intermediate" else 45
    if gym_days >= 5:
        session_minutes = max(20, session_minutes - 5)
    priorities = advice["Practice drill"].tolist()
    while len(priorities) < 3:
        priorities.append("Slow-motion rehearsal")
    return pd.DataFrame([
        {
            "Week": 1,
            "Focus": "Baseline and setup",
            "Sessions": 3,
            "Minutes per session": session_minutes,
            "Plan": f"{priorities[0]}; record one face-on test clip at the end of the week.",
        },
        {
            "Week": 2,
            "Focus": "Primary movement pattern",
            "Sessions": 3,
            "Minutes per session": session_minutes,
            "Plan": f"Continue {priorities[0]} and add {priorities[1]}; keep most swings below 70% effort.",
        },
        {
            "Week": 3,
            "Focus": "Sequence and variability",
            "Sessions": 3,
            "Minutes per session": session_minutes + 5,
            "Plan": f"Alternate {priorities[1]} with {priorities[2]}; test three clubs while preserving tempo.",
        },
        {
            "Week": 4,
            "Focus": "Transfer and reassessment",
            "Sessions": 2,
            "Minutes per session": session_minutes + 10,
            "Plan": "Use a random-target practice session, then upload a new clip and compare the same metrics.",
        },
    ])


class ContextualBanditCoach:
    """A small epsilon-greedy contextual bandit for per-weakness drill feedback."""

    def __init__(self, payload: Dict[str, Any] | None = None):
        self.q_values = {
            state: {action: 0.0 for action in actions}
            for state, actions in RL_ACTIONS.items()
        }
        self.counts = {
            state: {action: 0 for action in actions}
            for state, actions in RL_ACTIONS.items()
        }
        if payload:
            for state in self.q_values:
                self.q_values[state].update(payload.get("q_values", {}).get(state, {}))
                self.counts[state].update(payload.get("counts", {}).get(state, {}))

    def recommend(self, state: str, epsilon: float = 0.10, seed: int | None = None) -> str:
        available = RL_ACTIONS[state]
        rng = random.Random(seed) if seed is not None else random
        if rng.random() < epsilon:
            return rng.choice(available)
        return max(available, key=lambda action: (self.q_values[state][action], -available.index(action)))

    def update(self, state: str, action: str, rating_1_to_5: int) -> float:
        rating = int(clamp(int(rating_1_to_5), 1, 5))
        reward = (rating - 3) / 2.0
        self.counts[state][action] += 1
        count = self.counts[state][action]
        old_value = self.q_values[state][action]
        self.q_values[state][action] = old_value + (reward - old_value) / count
        return float(reward)

    def payload(self) -> Dict[str, Any]:
        return {"q_values": self.q_values, "counts": self.counts}

    def json_bytes(self) -> bytes:
        return json.dumps(self.payload(), indent=2).encode("utf-8")


def weakest_state(component_scores: Dict[str, float]) -> Tuple[str, float]:
    return min(component_scores.items(), key=lambda item: item[1])
