import gymnasium as gym
from gymnasium.envs.registration import register
import torch
from snake_ai.envs.snake_env import SnakeEnv
from snake_ai.models.dqn.double_dqn import DoubleDQN
from snake_ai.models.dqn.dqn import DQN
from snake_ai.models.dqn.estimartor import Estimator
from snake_ai.models.utils.epsilon_scheduler import get_epsilon_schedule
from snake_ai.models.utils.replay_memory import ReplayMemory
from snake_ai.models.utils.torch_wrapper import TorchWrapper

def train(agent: DQN, env: SnakeEnv, step_num: int = 100_000):
    
    stats, N = {"step_idx": [0], "ep_rewards": [0.0], "ep_steps": [0.0]}, 0

    (state, info), done = env.reset(), False
    for step in range(step_num):

        action = agent.step(state)
        
        # separate episode termination and episode truncation signals
        # is a very recent change in the Gym API. In Crafter, these two signals
        # are subsumed by `done`.
        state_, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        agent.learn(state, action, reward, state_, done)

        # some envs just update the state and are not returning a new one
        state = state_.clone()

        # stats
        stats["ep_rewards"][N] += reward
        stats["ep_steps"][N] += 1

        if done:
            # episode done, reset env!
            (state, info), done = env.reset(), False
        
            # some more stats
            if N % 10 == 0:
                print("[{0:3d}][{1:6d}], R/ep={2:6.2f}, steps/ep={3:2.0f}.".format(
                    N, step,
                    torch.tensor(stats["ep_rewards"][-10:]).mean().item(),
                    torch.tensor(stats["ep_steps"][-10:]).mean().item(),
                ))

            stats["ep_rewards"].append(0.0)  # reward accumulator for a new episode
            stats["ep_steps"].append(0.0)    # reward accumulator for a new episode
            stats["step_idx"].append(step)
            N += 1

    print("[{0:3d}][{1:6d}], R/ep={2:6.2f}, steps/ep={3:2.0f}.".format(
        N, step, torch.tensor(stats["ep_rewards"][-10:]).mean().item(),
        torch.tensor(stats["ep_steps"][-10:]).mean().item(),
    ))
    stats["agent"] = [agent.__class__.__name__ for _ in range(N+1)]

    torch.save(
        agent._estimator.state_dict(),
        'C:\\Users\\andre\\Desktop\\PersonalProjects\\SnakeAI\\runs\\last_model_dqn_bigger.pt'
    )
    return stats

if __name__ == "__main__":

    max_steps = 90000000
    gamma = 0.99
    lr = 1e-3
    seed = 13
    eval_episodes = 10
    grid_size = 7
    number_of_rewards = 1
    nsteps = 11
    device = torch.device('cuda:0')

    torch.manual_seed(seed)

    register(id="Snake-v0", entry_point="snake_ai.envs.snake_env:SnakeEnv")

    env = TorchWrapper(
        gym.make(
            "Snake-v0",
            render_mode="rgb_array",
            window_size=768,
            grid_size=grid_size,
            number_of_rewards=number_of_rewards,
            render_frame=False
        ),
        device=device
    )

    net = Estimator(
        in_channels=1,
        map_size=grid_size + 2,
        num_of_layers=2,
        channels=[2, 4],
        action_num=5,
        hidden_embedding_size=128,
        apply_pooling=False,
        padding=0,
        device=device,
    )

    stats = train(
        DoubleDQN(
            net,
            ReplayMemory(size=10000, batch_size=32, device=device),
            torch.optim.Adam(net.parameters(), lr=1e-3, eps=1e-4),
            get_epsilon_schedule(start=1.0, end=0.1, steps=1000000),
            action_num=5,
            warmup_steps=1000,
            update_steps=2,
            update_target_steps=256
        ),
        env,
        step_num=9000000  # change the experiment length if it's learning but not reaching about .95
    )