import numpy as np
from game.snake_env import SnakeEnv
from agents.dqn_agent import DQNAgent


def train_dqn(episodes: int = 5000, save_interval: int = 500,
              model_path: str = "dqn_model.pth"):
    env = SnakeEnv()
    agent = DQNAgent()
    
    scores = []
    best_avg = 0
    
    print("Starting DQN training...")
    print(f"Episodes: {episodes}")
    print(f"Device: {agent.device}")
    print(f"Initial epsilon: {agent.epsilon}")
    print("-" * 50)
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        step_count = 0
        while not done:
            action = agent.act(state, training=True)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            step_count += 1
            if step_count % 4 == 0:
                agent.train()
        
        score = env.get_score()
        scores.append(score)
        
        if (episode + 1) % 100 == 0:
            avg_score = np.mean(scores[-100:])
            max_score = max(scores[-100:]) if scores else 0
            
            if avg_score > best_avg:
                best_avg = avg_score
            
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
    train_dqn(episodes=5000, save_interval=500)
