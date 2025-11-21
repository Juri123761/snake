import argparse
import sys
import time
from game.snake_env import SnakeEnv
from graphics.snake_pygame import SnakeRenderer
from agents.q_learning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent


def _play_game(env, agent, agent_name: str):
    try:
        agent.epsilon = 0
        
        renderer = SnakeRenderer(width=env.width, height=env.height)
        
        print(f"Playing with {agent_name} agent...")
        print("Close the window to exit.")
        
        state = env.reset()
        done = False
        
        while not done:
            action = agent.act(state, training=False)
            state, reward, done = env.step(action)
            
            renderer.render(env.get_snake(), env.get_food(), env.get_score())
            
            if done:
                print(f"Game Over! Final Score: {env.get_score()}")
                time.sleep(2)
        
        renderer.close()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during gameplay: {e}")
        sys.exit(1)


def play_q_learning(model_path: str = "q_table.pkl"):
    env = SnakeEnv()
    agent = QLearningAgent()
    agent.load(model_path)
    _play_game(env, agent, "Q-learning")


def play_dqn(model_path: str = "dqn_model.pth"):
    env = SnakeEnv()
    agent = DQNAgent()
    agent.load(model_path)
    _play_game(env, agent, "DQN")


def main():
    parser = argparse.ArgumentParser(description="Play Snake with trained AI agent")
    parser.add_argument("--agent", type=str, choices=["qlearning", "dqn"],
                       default="qlearning", help="Agent type to use")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to model file (default: q_table.pkl or dqn_model.pth)")
    
    args = parser.parse_args()
    
    if args.agent == "qlearning":
        model_path = args.model or "q_table.pkl"
        play_q_learning(model_path)
    else:
        model_path = args.model or "dqn_model.pth"
        play_dqn(model_path)


if __name__ == "__main__":
    main()
