import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame

class Environment(gym.Env):
    def __init__(self, reward_version=5, max_steps=500, react_distance=8.0,
                 angle_strong_deg=25, road_width=20.0, verbose=False, seed=None):
        super().__init__()
        
        # --- Action space: [acceleration, steering angle] ---
        low_action = np.array([-2.0, -1.0], dtype=np.float32)
        high_action = np.array([2.0, 1.0], dtype=np.float32)
        self.action_space = spaces.Box(low=low_action, high=high_action, dtype=np.float32)

        # --- Observation space ---
        low_obs = np.array([-100, -100, 0, -np.pi, -15, -100, -100, -np.pi], dtype=np.float32)
        high_obs = np.array([100, 100, 20, np.pi, 15, 100, 100, np.pi], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # Obstacles
        self.obstacle_radius = 1.0
        self.obstacles = []

        # Vehicle
        self.vehicle_radius = 0.5
        self.prev_acceleration = 0.0

        # Road + reward config
        self.road_width = road_width
        self.reward_version = reward_version
        self.max_steps = max_steps
        self.react_distance = react_distance
        self.angle_strong = np.deg2rad(angle_strong_deg)
        self.verbose = verbose

        # RNG for reproducibility
        self.seed(seed)

    def seed(self, seed=None):
        self._seed = seed
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # optional, but ensures compatibility with gym.Env
        if seed is not None:
            self.seed(seed)
            
        self.x = 0
        self.y = 0
        self.prev_y = 0
        self.velocity = 0
        self.heading = np.pi / 2
        self.center_offset = 0
        self.step_count = 0

        # Place obstacles ahead
        # Sample from normal and round and clip
        # self.num_obstacles = int(np.clip(round(np.random.normal(loc=16, scale=3)), 8, 25)) #clipping to range [8,22] for no. of obstscles generated
        self.num_obstacles = 30
        self.obstacles = []

        min_distance = 2 * self.obstacle_radius + 0.1  # add buffer to ensure spacing

        attempts = 0
        max_attempts = 1000  # safety cap to prevent infinite loops

        while len(self.obstacles) < self.num_obstacles and attempts < max_attempts:
            obs_x = self.np_random.uniform(-self.road_width / 2 + 1, self.road_width / 2 - 1)
            obs_y = self.np_random.uniform(self.y + 20, self.y + 400)
            
            too_close = False
            for obs in self.obstacles:
                dx = obs_x - obs['x']
                dy = obs_y - obs['y']
                dist_sq = dx**2 + dy**2
                if dist_sq < min_distance**2:
                    too_close = True
                    break

            if not too_close:
                self.obstacles.append({'x': obs_x, 'y': obs_y})
            
            attempts += 1

        if self.verbose and attempts >= max_attempts:
            print(f"Warning: Obstacle generation stopped after {attempts} attempts. Placed {len(self.obstacles)} obstacles.")

        self.obs_dx, self.obs_dy, self.rel_angle = self.get_relative_obstacle_info()

        state = np.array([
            self.x, self.y, self.velocity, self.heading,
            self.center_offset, self.obs_dx, self.obs_dy, self.rel_angle
        ], dtype=np.float32)

        return state, {}

    def step(self, action):
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            if self.verbose:
                print(f"Invalid action detected: {action}")
            action = np.zeros_like(action)

        action = np.clip(action, self.action_space.low, self.action_space.high)
        acceleration, angle = action
        dt = 0.1

        self.velocity += acceleration * dt
        self.velocity = np.clip(self.velocity, 0, 6)
        self.heading += angle * dt
        self.heading = (self.heading + np.pi) % (2 * np.pi) - np.pi

        self.x += self.velocity * np.cos(self.heading) * dt
        self.y += self.velocity * np.sin(self.heading) * dt

        self.center_offset = self.x
        self.step_count += 1

        self.obs_dx, self.obs_dy, self.rel_angle = self.get_relative_obstacle_info()
        obs_dist = np.sqrt(self.obs_dx ** 2 + self.obs_dy ** 2) + 1e-6

        # --- Dynamic reward ---
        progress = self.step_count / self.max_steps
        w_center = 0.15 + 0.1 * progress
        w_heading = 0.7 + 0.7 * progress
        w_forward = 3.0 + 2.0 * progress
        w_obstacle = 2.5 - 1.0 * progress
        w_turning = 0.05
        w_accel_change = 0.4

        if obs_dist < self.react_distance:
            w_center *= obs_dist / self.react_distance

        p_off_center = -w_center * abs(self.center_offset)
        r_center_bonus = max(0.0, 1.0 - abs(self.center_offset) / 5.0)
        p_heading = -w_heading * abs(self.heading - np.pi / 2)
        p_turning = -w_turning * abs(angle)

        rel_angle_abs = abs(self.rel_angle)
        distance_factor = max(0.0, 1.0 - obs_dist / self.react_distance)
        p_obstacle = -w_obstacle * distance_factor if rel_angle_abs < self.angle_strong else 0.0

        accel_change = acceleration - self.prev_acceleration
        self.prev_acceleration = acceleration
    
        p_accel_change = -w_accel_change * abs(accel_change)

        delta_y = self.y - self.prev_y

        # Checking if heading is aligned within a tolerance (within 45 degrees of straight up)
        heading_error = abs(self.heading - np.pi/2)
        aligned = heading_error < np.deg2rad(45)

        # Reward only forward progress in aligned direction, else punish
        if delta_y > 0 and aligned:
            r_forward = w_forward * delta_y
        else:
            r_forward = -w_forward * abs(delta_y)

        self.prev_y = self.y

        if self.reward_version == 1:
            reward = 0.0
        elif self.reward_version == 2:
            reward = r_forward
        elif self.reward_version == 3:
            reward = r_forward + p_off_center + r_center_bonus
        elif self.reward_version == 4:
            reward = r_forward + p_off_center + r_center_bonus + p_obstacle + p_accel_change 
        elif self.reward_version == 5:
            reward = (
                p_off_center +
                r_center_bonus +
                p_heading +
                p_turning +
                p_obstacle +
                p_accel_change +
                r_forward
            )
        else:
            raise ValueError(f"Unknown reward version: {self.reward_version}")

        info = {"reason": "running"}
        terminated, truncated = False, False

        if any(
            np.sqrt((obs['x'] - self.x) ** 2 + (obs['y'] - self.y) ** 2) <= self.vehicle_radius + self.obstacle_radius
            for obs in self.obstacles
        ):
            reward -= 20
            info["reason"] = "collision"
            terminated = True

        elif abs(self.center_offset) > self.road_width / 2:
            reward -= 20
            info["reason"] = "offroad"
            terminated = True

        elif self.step_count >= self.max_steps:
            truncated = True
            info["reason"] = "timeout"

        state = np.array([
            self.x, self.y, self.velocity, self.heading,
            self.center_offset, self.obs_dx, self.obs_dy, self.rel_angle
        ], dtype=np.float32)

        return state, reward, terminated, truncated, info

    def get_relative_obstacle_info(self):
        if not self.obstacles:
            return 0.0, 0.0, 0.0

        min_dist = float('inf')
        nearest_dx = nearest_dy = 0.0
        for obstacle in self.obstacles:
            dx = obstacle["x"] - self.x
            dy = obstacle["y"] - self.y
            dist = np.sqrt(dx ** 2 + dy ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest_dx = dx
                nearest_dy = dy

        rel_angle = np.arctan2(nearest_dy, nearest_dx) - self.heading
        rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi

        return nearest_dx, nearest_dy, rel_angle

    def render(self, mode="human"):
        render_scale = 16  # 1 unit = 16 pixels

        if not hasattr(self, "render_initialized") or not self.render_initialized:
            pygame.init()
            self.screen_width = 800
            self.screen_height = 600
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Self-Driving Bike - PPO")
            self.clock = pygame.time.Clock()

            # Load and resize images according to physical radius
            raw_bike_img = pygame.image.load("ReinforcementLearning/Assets/Bike.png").convert_alpha()
            raw_car_img = pygame.image.load("ReinforcementLearning/Assets/Car.png").convert_alpha()

            bike_diameter = int(2 * self.vehicle_radius * render_scale *1.4)
            car_diameter = int(2 * self.obstacle_radius * render_scale *1.4)

            self.bike_img = pygame.transform.scale(raw_bike_img, (bike_diameter, bike_diameter * 2))
            self.car_img = pygame.transform.scale(raw_car_img, (car_diameter, car_diameter * 2))

            self.render_initialized = True

        self.screen.fill((50, 50, 50))  # dark gray background (road)

        # Camera tracks bike's y-position
        camera_offset_y = self.y - self.screen_height / render_scale / 2

        # Draw road
        road_left = self.screen_width // 2 - int(self.road_width * render_scale / 2)
        road_right = self.screen_width // 2 + int(self.road_width * render_scale / 2)
        pygame.draw.rect(self.screen, (100, 100, 100), (road_left, 0, road_right - road_left, self.screen_height))

        # Draw obstacles (cars)
        for obs in self.obstacles:
            screen_x = int(self.screen_width / 2 + obs['x'] * render_scale)
            screen_y = int(self.screen_height - (obs['y'] - camera_offset_y) * render_scale)
            self.screen.blit(self.car_img, self.car_img.get_rect(center=(screen_x, screen_y)))

        # Draw agent (bike)
        bike_x = int(self.screen_width / 2 + self.x * 10)
        bike_y = int(self.screen_height - (self.y - camera_offset_y) * render_scale)
        angle_deg = -np.degrees(self.heading) + 90
        bike_rotated = pygame.transform.rotate(self.bike_img, angle_deg)
        rect = bike_rotated.get_rect(center=(bike_x, bike_y))
        self.screen.blit(bike_rotated, rect)

        pygame.display.flip()
        self.clock.tick(30)
        pygame.event.pump()

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                exit()


    def close(self):
        pass
