import os
import json
from unityagents import UnityEnvironment

import session
from noise import OrnsteinUhlenbeckNoise
from experience import ReplayBuffer
from agent import DDPGAgent
import utils


def main():
    with open(os.path.join(".", "configs", "reacher_one.json"), "r") as read_file:
        config = json.load(read_file)

    env = UnityEnvironment(file_name=os.path.join(*config["env_path"]))

    noise = OrnsteinUhlenbeckNoise(config["n_actions"], config["mu"], config["theta"], config["sigma"], config["seed"])
    replay_buffer = ReplayBuffer(config["buffer_size"], config["device"], config["seed"])

    agent = DDPGAgent(config, noise, replay_buffer)

    if config["run_training"]:
        session.train(agent, env, config)
        checkpoint_dir = os.path.join(".", *config["checkpoint_dir"], config["env_name"])
        utils.save_state_dict(os.path.join(checkpoint_dir, "actor"), agent.actor.state_dict())
        utils.save_state_dict(os.path.join(checkpoint_dir, "critic"), agent.critic.state_dict())
    else:
        checkpoint_dir = os.path.join(".", *config["checkpoint_dir"], config["env_name"])
        agent.actor.load_state_dict(utils.load_latest_available_state_dict(os.path.join(checkpoint_dir, "actor", "*")))
        agent.critic.load_state_dict(utils.load_latest_available_state_dict(os.path.join(checkpoint_dir, "critic", "*")))
        session.evaluate(agent, env, num_test_runs=1)

    env.close()


if __name__ == '__main__':
    main()
