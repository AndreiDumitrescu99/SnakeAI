import torch
import torch.optim as optim
from torchinfo import summary

from gymnasium.envs.registration import register
import gymnasium as gym
from snake_ai.models.a2c.actor_critic import A2C
from snake_ai.models.a2c.policy import ActorCriticPolicy
from snake_ai.envs.snake_env import SnakeEnv
from snake_ai.models.utils.torch_wrapper import TorchWrapper

def eval_loop(
    agent: A2C,
    env: SnakeEnv,
    eval_episodes: int,
) -> None:
    
    episodic_returns = []

    for _ in range(eval_episodes):
        state, done = env.reset()[0].clone(), False
        episodic_returns.append(0)

        while not done:
            if len(state.shape) == 3:
                state = torch.unsqueeze(state, 0)

            action = agent.best_act(state)
            # print(state, action)
            state, reward, done, _, _ = env.step(action)
            episodic_returns[-1] += reward
    
if __name__ == "__main__":

    gamma = 0.99
    grid_size = 8
    path_to_save_model = 'C:\\Users\\andre\\Desktop\\PersonalProjects\\SnakeAI\\runs\\best_model_small.pt'
    device = torch.device('cuda:0')

    register(id="Snake-v0", entry_point="snake_ai.envs.snake_env:SnakeEnv")
    env = TorchWrapper(
        gym.make(
            "Snake-v0",
            render_mode="rgb_array",
            window_size=768,
            grid_size=grid_size,
            number_of_rewards=1,
            render_frame=False
        ),
        device=device
    )

    policy = ActorCriticPolicy(
        in_channels=1,
        map_size=grid_size + 2,
        num_of_layers=3,
        channels=[1, 2, 4],
        action_num=5,
        hidden_embedding_size=128,
        apply_pooling=False,
        device=device,
    ).to(device)
    print(policy.embed_size)
    print(summary(policy, (1, 10, 10)), policy)
    checkpoint = torch.load(path_to_save_model)
    print(checkpoint)
    policy.load_state_dict(checkpoint)

    agent = A2C(
        policy=policy,
        gamma=gamma,
        optimizer=optim.Adam(policy.parameters(), lr=1e-3, eps=1e-05),
        nsteps=5,
        device=device
    )

    eval_loop(
        agent,
        env,
        10
    )