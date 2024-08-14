import torch
import pygame
import random
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim

GRID_SIZE = 50
PIXEL_SIZE = 15
FOOD_REWARD = 100
MOVE_PENALTY = -0.1
CLOSER_REWARD = 1
FURTHER_PENALTY = -1
DEATH_PENALTY = -50
TIME_PENALTY = -0.1
SURVIVAL_BONUS = 0.1
MIN_EPSILON = 0.25
FPS = 40  # Adjust FPS for game speed
EXPERIENCE_REPLAY_SIZE = 10000
BATCH_SIZE = 128
GAMMA = 0.99
MODEL_SAVE_INTERVAL = 2000
LEARNING_RATE = 0.0001
EPISODES = 1000000  # Set the number of episodes for training
STUCK_THRESHOLD = 1000  # Adjust this threshold as needed

# Colors
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

class DQN(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class DQNAgent:
    def __init__(self, input_dim, action_dim, device):
        self.device = device
        self.action_dim = action_dim
        self.q_network = DQN(input_dim, action_dim).to(device)
        self.target_network = DQN(input_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = deque(maxlen=EXPERIENCE_REPLAY_SIZE)
        self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def store_experience(self, experience):
        self.replay_buffer.append(experience)

    def train(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return
    
        batch = random.sample(self.replay_buffer, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
    
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()

class SnakeGame:
    def __init__(self, render=True):
        self.grid_size = GRID_SIZE
        self.pixel_size = PIXEL_SIZE
        self.render_enabled = render
        if self.render_enabled:
            self.screen = pygame.display.set_mode((self.grid_size * self.pixel_size, self.grid_size * self.pixel_size))
            pygame.display.set_caption('Improved Snake Game with DQN')
        self.reset()
        self.clock = pygame.time.Clock()  # Add this line for controlling FPS

    def place_food(self):
        while True:
            self.food = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            if self.food not in self.snake:
                break

    def reset(self):
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = (0, 1)  # Initial direction is moving right
        self.place_food()
        self.score = 0
        self.steps_since_food = 0
        self.steps_since_last_progress = 0  # Initialize here
        self.previous_distance = self.get_distance(self.snake[0], self.food)
        self.previous_positions = deque(maxlen=5)  # Track the last 5 positions
        return self.get_state()

    def get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food

        distance_to_food = self.get_distance(self.snake[0], self.food)
        food_left = int(food_x < head_x)
        food_right = int(food_x > head_x)
        food_up = int(food_y < head_y)
        food_down = int(food_y > head_y)

        distance_change = distance_to_food - self.previous_distance

        direction_left = int((head_x - 1, head_y) in self.snake or head_x - 1 < 0)
        direction_right = int((head_x + 1, head_y) in self.snake or head_x + 1 >= self.grid_size)
        direction_up = int((head_x, head_y - 1) in self.snake or head_y - 1 < 0)
        direction_down = int((head_x, head_y + 1) in self.snake or head_y + 1 >= self.grid_size)

        state = (
            head_x, head_y, food_x, food_y,
            self.direction[0], self.direction[1],
            direction_left, direction_right, direction_up, direction_down,
            food_left, food_right, food_up, food_down,
            distance_change
        )
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        self.change_direction(action)
        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])

        reward = MOVE_PENALTY
        done = False

        if self.is_collision(new_head):
            reward = DEATH_PENALTY
            done = True
        else:
            old_distance = self.previous_distance
            self.snake = [new_head] + self.snake[:-1]
            new_distance = self.get_distance(self.snake[0], self.food)
            self.previous_distance = new_distance
            self.steps_since_food += 1

            if new_head == self.food:
                self.snake.append(self.snake[-1])
                self.score += 1  # This line updates the score
                reward = FOOD_REWARD
                self.place_food()
                self.steps_since_food = 0
                self.previous_distance = self.get_distance(self.snake[0], self.food)

            else:
                if new_distance < old_distance:
                    reward += CLOSER_REWARD
                else:
                    reward += FURTHER_PENALTY

            reward += SURVIVAL_BONUS - (self.steps_since_food * TIME_PENALTY)

            # Check if stuck
            if self.steps_since_last_progress >= STUCK_THRESHOLD:
                done = True
            else:
                # Update the steps since last progress
                if new_distance != old_distance:
                    self.steps_since_last_progress = 0
                else:
                    self.steps_since_last_progress += 1

        return self.get_state(), reward, done

    def change_direction(self, action):
        left_turns = {
            (1, 0): (0, 1),   # Moving right, turn left -> down
            (0, 1): (-1, 0),  # Moving down, turn left -> left
            (-1, 0): (0, -1), # Moving left, turn left -> up
            (0, -1): (1, 0)   # Moving up, turn left -> right
        }
        
        right_turns = {
            (1, 0): (0, -1),  # Moving right, turn right -> up
            (0, 1): (1, 0),   # Moving down, turn right -> right
            (-1, 0): (0, 1),  # Moving left, turn right -> down
            (0, -1): (-1, 0)  # Moving up, turn right -> left
        }
        
        # Determine new direction
        if action == 0:  # Turn left
            new_direction = left_turns.get(self.direction, self.direction)
        elif action == 1:  # Go straight
            new_direction = self.direction
        elif action == 2:  # Turn right
            new_direction = right_turns.get(self.direction, self.direction)
        else:
            print("Invalid action")
            return
        
        # Check for reversing direction
        if (new_direction[0] == -self.direction[0] and new_direction[1] == -self.direction[1]):
            print("Reversing direction not allowed")
            return
        
        self.direction = new_direction

    def is_collision(self, position):
        x, y = position
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        if position in self.snake:
            return True
        return False

    def get_distance(self, position1, position2):
        return abs(position1[0] - position2[0]) + abs(position1[1] - position2[1])

    def render(self):
        if not self.render_enabled:
            return
        self.screen.fill(BLACK)
        for segment in self.snake:
            pygame.draw.rect(self.screen, GREEN, pygame.Rect(segment[0] * self.pixel_size, segment[1] * self.pixel_size, self.pixel_size, self.pixel_size))
        pygame.draw.rect(self.screen, RED, pygame.Rect(self.food[0] * self.pixel_size, self.food[1] * self.pixel_size, self.pixel_size, self.pixel_size))
        pygame.display.flip()

def run_inference(num_episodes=50, render=True):
    pygame.init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 15
    action_dim = 3
    agent = DQNAgent(input_dim, action_dim, device)
    game = SnakeGame(render=render)

    # Load the trained model
    model_path = "best_snake_dqn_model.pth"
    try:
        agent.q_network.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        print(f"Model file {model_path} not found.")
        pygame.quit()
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        pygame.quit()
        return

    agent.q_network.eval()

    scores = []

    try:
        for episode in range(num_episodes):
            state = game.reset()
            done = False
            total_reward = 0

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = agent.q_network(state_tensor)
                action = agent.select_action(state)  # Select action without epsilon

                next_state, reward, done = game.step(action)
                state = next_state
                total_reward += reward

                if render:
                    game.render()

                game.clock.tick(FPS)  # Control the frame rate

            scores.append(game.score)
            print(f"Episode {episode + 1}/{num_episodes} - Score: {game.score}, Total Reward: {total_reward:.2f}")

    finally:
        pygame.quit()

    print(f"Average Score over {num_episodes} episodes: {sum(scores) / num_episodes:.2f}")
    print(f"Highest Score in {num_episodes} episodes: {max(scores)}")

if __name__ == "__main__":
    run_inference(num_episodes=50, render=True)