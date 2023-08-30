import torch
import gymnasium as gym
from gymnasium.envs.registration import register
from snake_ai.models.dqn.double_dqn import DoubleDQN
from snake_ai.models.dqn.dqn import DQN
from snake_ai.envs.snake_env import SnakeEnv
from snake_ai.models.utils.epsilon_scheduler import get_epsilon_schedule
from snake_ai.models.dqn.estimartor import Estimator
from snake_ai.models.utils.replay_memory import ReplayMemory
from snake_ai.models.utils.torch_wrapper import TorchWrapper

def eval(agent: DQN, env: SnakeEnv, eval_episodes: int = 10):

    for _ in range(eval_episodes):

        (state, info), done = env.reset(), False

        while not done: 
            action = agent.best_act(state)

            state, _, done, _, _ = env.step(action)
            print(state, state.shape)


if __name__ == "__main__":

    seed = 13
    eval_episodes = 10
    grid_size = 7
    snake_length = 5
    number_of_rewards = 1
    path_to_save_model = 'C:\\Users\\andre\\Desktop\\PersonalProjects\\SnakeAI\\runs\\best_model_dqn_bigger.pt'
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
            render_frame=True,
            snake_length = snake_length
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

    checkpoint = torch.load(path_to_save_model)
    # print(checkpoint)
    net.load_state_dict(checkpoint)

    stats = eval(
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
    )