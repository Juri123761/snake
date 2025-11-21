import numpy as np
from game.snake_env import SnakeEnv
from agents.dqn_agent import DQNAgent


def train_dqn(episodes: int = 15000, save_interval: int = 1000,
              model_path: str = "dqn_model.pth"):
    env = SnakeEnv()
    agent = DQNAgent()
    
    scores = []
    rewards_history = []
    best_avg = 0
    
    print("Starting DQN training...")
    print(f"Episodes: {episodes}")
    print(f"Device: {agent.device}")
    print(f"Initial epsilon: {agent.epsilon}")
    print("-" * 50)
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state, training=True)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            agent.train()
        
        score = env.get_score()
        scores.append(score)
        rewards_history.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            avg_score = np.mean(scores[-100:])
            avg_reward = np.mean(rewards_history[-100:])
            max_score = max(scores[-100:]) if scores else 0
            best_avg = max(best_avg, avg_score)
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Avg Score: {avg_score:.2f} | "
                  f"Max Score: {max_score} | "
                  f"Best Avg: {best_avg:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Memory: {len(agent.memory)}")
        
        if (episode + 1) % save_interval == 0:
            try:
                agent.save(model_path)
                print(f"Model saved to {model_path}")
            except Exception as e:
                print(f"Warning: Failed to save model: {e}")
    
    try:
        agent.save(model_path)
        print(f"\nTraining completed! Final model saved to {model_path}")
    except Exception as e:
        print(f"Warning: Failed to save final model: {e}")
    
    if len(scores) >= 100:
        print(f"Final average score (last 100 episodes): {np.mean(scores[-100:]):.2f}")
        print(f"Best average score: {best_avg:.2f}")
    else:
        print(f"Final average score: {np.mean(scores):.2f}")


if __name__ == "__main__":
    train_dqn(episodes=15000, save_interval=1000)
