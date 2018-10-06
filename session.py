import numpy as np


def test(agent, env, num_test_runs=3):
    brain_name = env.brain_names[0]

    for episode in range(num_test_runs):
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        while True:
            actions = np.random.randn(1, 4)  # select an action (for each agent)
            actions = np.clip(actions, -1, 1)  # all actions between -1 and 1
            env_info = env.step(actions)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            score += reward
            state = next_state
            if done:
                break

    print("Score at Episode {}: {}".format(episode, score))
