import numpy as np
from game.snake_env import SnakeEnv
from agents.q_learning_agent import QLearningAgent


def train_q_learning(episodes: int = 1000, save_interval: int = 100,
                    model_path: str = "q_table.pkl"):
    env = SnakeEnv()
    agent = QLearningAgent()
    
    scores = []
    rewards_history = []
    
    print("Starting Q-learning training...")
    print(f"Episodes: {episodes}")
    print(f"Initial epsilon: {agent.epsilon}")
    print("-" * 50)
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state, training=True)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        
        score = env.get_score()
        scores.append(score)
        rewards_history.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            avg_score = np.mean(scores[-100:])
            avg_reward = np.mean(rewards_history[-100:])
            max_score = max(scores[-100:]) if scores else 0
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Avg Score: {avg_score:.2f} | "
                  f"Max Score: {max_score} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
        
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
    else:
        print(f"Final average score: {np.mean(scores):.2f}")


if __name__ == "__main__":
    train_q_learning(episodes=1000, save_interval=100)
