import numpy as np
import protocol


def _run_train_episode(agent, env, brain_name, train_log):
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    while True:
        action = agent.act_noisy(state)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]
        agent.save_experiences(state, action, reward, next_state, done)
        actor_loss, critic_loss = agent.learn()
        state = next_state
        train_log.step_update(reward, actor_loss, critic_loss)
        if done:
            break


def train(agent, env, config):
    # get the default brain
    brain_name = env.brain_names[0]
    train_log = protocol.Protocol(config)

    for i_episode in range(1, config["max_num_episodes"] + 1):
        _run_train_episode(agent, env, brain_name, train_log)
        train_log.episode_update()
        train_log.write_statistics()
        print("\r", train_log, "\n", end="")

        mean_reward = np.mean(np.mean(train_log.episode_rewards[-100:]))
        if mean_reward >= config["average_score_for_solving"]:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100, mean_reward))
            break


def evaluate(agent, env, num_test_runs=3):
    brain_name = env.brain_names[0]

    for episode in range(num_test_runs):
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        while True:
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            score += reward
            state = next_state
            if done:
                break

        print("Score at Episode {}: {}".format(episode, score))
