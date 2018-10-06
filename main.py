import os
import json
from unityagents import UnityEnvironment
import session


def main():
    with open(os.path.join(".", "configs", "reacher_one.json"), "r") as read_file:
        config = json.load(read_file)

    env = UnityEnvironment(file_name=os.path.join(*config["env_path"]))

    if config["run_training"]:
        pass
    else:
        session.test(None, env, 1)

    env.close()


if __name__ == '__main__':
    main()
