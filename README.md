Welcome to Snake AI:

The code uses DQN reinforcement learning.


The input dimensions for your DQN model are determined by the state representation you provide to the model. Based on your code, the state is represented by the following features:

Normalized Snake Head Position: (2 features - head_x, head_y)
Normalized Food Position: (2 features - food_x, food_y)
Direction of the Snake: (2 features - direction_x, direction_y)
Distance Change to Food: (1 feature - distance_change)
Distances to Walls: (4 features - distance_to_left_wall, distance_to_right_wall, distance_to_top_wall, distance_to_bottom_wall)
Snake Length: (1 feature - snake_length)
Average Distance to Food: (1 feature - avg_distance_to_food)
Steps Since Last Food: (1 feature - steps_since_last_food)
Current Score: (1 feature - current_score)
Relative Angle to Food: (1 feature - angle)
Body Segment Information: Each body segment includes:
Relative position (2 features - rel_x, rel_y)
Distance to segment (1 feature - distance_to_segment)
Angle to segment (1 feature - segment_angle)
You process up to 50 body segments, which results in 
50×4=200 features.

Adding these together:

Snake head position: 2

Food position: 2

Direction: 2

Distance change: 1

Distances to walls: 4

Snake length: 1

Average distance to food: 1

Steps since last food: 1

Current score: 1

Relative angle to food: 1

Body segment information: 200

Total input dimensions = 216

Thus, the input dimension for your DQN model is 216.
