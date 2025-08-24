import os
from crypto_rl.strategic_agent import train_strategic_agent


def main():
    out = os.path.join("output")
    os.makedirs(out, exist_ok=True)
    train_strategic_agent(episodes=3000, eval_every=200, out_dir=out)


if __name__ == "__main__":
    main()
