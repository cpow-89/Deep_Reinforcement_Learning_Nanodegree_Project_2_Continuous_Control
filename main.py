import os
import json
import session
import torch
from unityagents import UnityEnvironment


def main():
    with open(os.path.join(".", "configs", "reacher_one.json"), "r") as read_file:
        config = json.load(read_file)

    env = UnityEnvironment(file_name=os.path.join(*config["env_path"]))
    device = torch.device(config["device"])

    if config["run_training"]:
        pass
    else:
        session.test(None, env, 1)

    env.close()


if __name__ == '__main__':
    main()
