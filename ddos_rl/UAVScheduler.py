import os
from .agent import QLearningAgent
from .profiles import CPU_FREQUENCY_PRESETS

QTABLE_CANDIDATES = [
    os.path.join("output_smoke", "tactical_q_table_best.npy"),
    os.path.join("output_smoke", "tactical_q_table.npy"),
    "tactical_q_table_best.npy",
    "tactical_q_table.npy",
]


class UAVScheduler:
    def __init__(self, qtable_path: str | None = None):
        self.state_dims = [4, 4, 3, 3]
        self.action_dim = 9
        self.agent = QLearningAgent(self.state_dims, self.action_dim)

        path = qtable_path
        if path is None:
            for cand in QTABLE_CANDIDATES:
                if os.path.exists(cand):
                    path = cand
                    break
        if path is None:
            raise FileNotFoundError(
                "No trained tactical Q-table found. Run train_tactical.py to generate one."
            )

        self.agent.load_policy(path)
        print(f"Using Q-table: {path}")

        self._freq_keys = list(CPU_FREQUENCY_PRESETS.keys())

        self._scenarios = [
            [0, 3, 0, 2],
            [1, 2, 1, 1],
            [2, 1, 2, 0],
            [3, 0, 2, 0],
            [1, 3, 1, 2],
        ]
        self._scenario_idx = 0

    def _decode_action(self, action: int) -> str:
        if action == 8:
            return "DE-ESCALATE"
        model = "XGBOOST" if (action // 4) == 0 else "TST"
        freq_idx = action % 4
        preset = self._freq_keys[freq_idx]
        return f"{model}@{preset}"

    def _get_simulated_state(self):
        state = self._scenarios[self._scenario_idx % len(self._scenarios)]
        self._scenario_idx += 1
        return state

    def _manage_tactical_policy_rl(self, state):
        action = self.agent.choose_action(state, training=False)
        readable = self._decode_action(action)
        print(f"[Policy] State {state} -> Action {action} ({readable})")
        return readable

    def run(self, steps=5):
        print("=== UAVScheduler (Tactical) Demo ===")
        print("State format: [Threat, Battery, CPU Load, Task Priority]")
        for i in range(steps):
            print(f"\n--- Step {i+1} ---")
            state = self._get_simulated_state()
            self._manage_tactical_policy_rl(state)


if __name__ == "__main__":
    scheduler = UAVScheduler()
    scheduler.run(steps=5)