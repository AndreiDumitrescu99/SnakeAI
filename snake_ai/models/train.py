import torch
import torch.optim as optim
from gymnasium.envs.registration import register
import gymnasium as gym

from typing import List

from snake_ai.models.a2c.actor_critic import A2C
from snake_ai.models.a2c.policy import ActorCriticPolicy
from snake_ai.envs.snake_env import SnakeEnv
from snake_ai.models.utils.torch_wrapper import TorchWrapper

LOG_INTERVAL = 1000
max_reward = 0

def _save_stats(
    episodic_returns: List[int],
    crt_step: int,
) -> float:

    # save the evaluation stats
    global max_reward
    episodic_returns = torch.tensor(episodic_returns)
    avg_return = episodic_returns.mean().item()
    print(
        "[{:07d}] eval results: R/ep={:03.2f}, std={:03.2f}, Max Reward={:03.2f}.".format(
            crt_step, avg_return, episodic_returns.std().item(), max_reward
        )
    )
    
    return avg_return

def eval_loop(
    agent: A2C,
    env: SnakeEnv,
    crt_step: int,
    eval_episodes: int,
    path_to_save_model: str
) -> None:
    
    episodic_returns = []
    global max_reward

    for _ in range(eval_episodes):
        state, done = env.reset()[0].clone(), False
        episodic_returns.append(0)

        while not done:
            if len(state.shape) == 3:
                state = torch.unsqueeze(state, 0)

            action = agent.best_act(state)
            state, reward, done, _, _ = env.step(action)
            episodic_returns[-1] += reward
    
    avg_return = _save_stats(episodic_returns, crt_step)

    if avg_return > max_reward:
        max_reward = avg_return
        torch.save(
            agent._policy.state_dict(),
            path_to_save_model
        )

def train_loop(
    agent: A2C,
    env: SnakeEnv,
    eval_env: SnakeEnv,
    max_steps: int,
    eval_episodes: int,
    path_to_save_model: str,
    device: torch.device
) -> None:
    
    agent_id = type(agent).__name__

    step, ep_cnt = 0, 0

    state, done = env.reset()[0].clone(), False

    while step < max_steps or not done:

        if done:
            ep_cnt += 1
            state, done = env.reset()[0].clone(), False
        
        if len(state.shape) == 3:
            state = torch.unsqueeze(state, 0)
        
        state = state.to(device=device)
        action = agent.act(state)
        state_, reward, done, _, _ = env.step(action=action)
        agent.learn(reward, state_.to(device), done)

        state = state_.clone()

        step += 1

        if step % LOG_INTERVAL == 0:
            eval_loop(agent, eval_env, step, eval_episodes, path_to_save_model)

if __name__ == "__main__":

    # TODO: Remove hardcodings. This is just for test.
    max_steps = 1000000
    gamma = 0.99
    lr = 1e-4
    seed = 13
    eval_episodes = 10
    grid_size = 16
    path_to_save_model = 'C:\\Users\\andre\\Desktop\\PersonalProjects\\SnakeAI\\runs\\best_model.pt'
    device = torch.device('cuda:0')

    torch.manual_seed(seed)

    register(id="Snake-v0", entry_point="snake_ai.envs.snake_env:SnakeEnv")
    env = TorchWrapper(
        gym.make(
            "Snake-v0",
            render_mode="rgb_array",
            window_size=768,
            grid_size=grid_size
        ),
        device=device
    )
    eval_env = TorchWrapper(
        gym.make(
            "Snake-v0",
            render_mode="rgb_array",
            window_size=768,
            grid_size=grid_size,
            render_frame=False
        ),
        device=device
    )

    policy = ActorCriticPolicy(map_size=grid_size, device=device).to(device)
    agent = A2C(
        policy=policy,
        gamma=gamma,
        optimizer=optim.Adam(policy.parameters(), lr=lr, eps=1e-05),
        nsteps=8,
        device=device
    )

    stats = train_loop(
        agent,
        env,
        eval_env,
        max_steps = max_steps,
        eval_episodes = eval_episodes,
        path_to_save_model = path_to_save_model,
        device=device
    )
