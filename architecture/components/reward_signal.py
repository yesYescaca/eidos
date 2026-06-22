"""Intrinsic reward — surprise reduction and goal achievement."""

from __future__ import annotations

from collections import deque


class RewardSignal:
    """Reward = prediction error reduction (intrinsic) + goal achievement (extrinsic)."""

    def __init__(self, history_window: int = 50) -> None:
        self.history_window = history_window
        self._error_history: deque[float] = deque(maxlen=history_window)
        self._reward_history: list[float] = []
        self._current_error: float = 0.0
        self._previous_error: float = 0.0
        self._goal_achieved: bool = False

    def intrinsic_reward(self) -> float:
        """Positive when surprise decreases."""
        return self._previous_error - self._current_error

    def extrinsic_reward(self, achieved: bool | None = None) -> float:
        flag = self._goal_achieved if achieved is None else achieved
        return 10.0 if flag else 0.0

    def total_reward(self) -> float:
        return self.intrinsic_reward() + self.extrinsic_reward()

    def update(self, new_error: float, goal_achieved: bool = False) -> float:
        self._previous_error = self._current_error if self._error_history else new_error
        self._current_error = new_error
        self._goal_achieved = goal_achieved
        self._error_history.append(new_error)
        reward = self.total_reward()
        self._reward_history.append(reward)
        return reward

    def get_reward_history(self) -> list[float]:
        return list(self._reward_history)

    def to_dict(self) -> dict:
        return {
            "error_history": list(self._error_history),
            "reward_history": self._reward_history,
            "current_error": self._current_error,
            "previous_error": self._previous_error,
        }

    def load_dict(self, data: dict) -> None:
        self._error_history = deque(data.get("error_history", []), maxlen=self.history_window)
        self._reward_history = data.get("reward_history", [])
        self._current_error = data.get("current_error", 0.0)
        self._previous_error = data.get("previous_error", 0.0)
