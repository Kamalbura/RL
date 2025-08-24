import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from crypto_rl.strategic_agent import StrategicCryptoEnv, StrategicCryptoAgent


def test_env_step_and_agent_choice():
    env = StrategicCryptoEnv()
    agent = StrategicCryptoAgent()
    s = env.reset()
    a = agent.choose_action(s, training=False)
    ns, r, done, info = env.step(a)
    assert isinstance(ns, list) and len(ns) == 3
    assert isinstance(r, float)
    assert isinstance(done, bool)
    assert "algorithm" in info
