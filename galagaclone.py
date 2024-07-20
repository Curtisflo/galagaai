import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import tqdm
import math

# Initialize Pygame
pygame.init()

# Set up the game window
WIDTH = 800
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Galaga Clone with AI")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Player
class Player(pygame.sprite.Sprite):
    def __init__(self, game_state):
        super().__init__()
        self.game_state = game_state
        self.image = pygame.Surface((50, 50))
        self.image.fill(GREEN)
        self.rect = self.image.get_rect()
        self.rect.centerx = WIDTH // 2
        self.rect.bottom = HEIGHT - 10
        self.speed = 5
        self.ai_controlled = True

    def update(self, action=None):
        if self.ai_controlled and action is not None:
            if action == 0:  # Move left
                self.rect.x -= self.speed
            elif action == 1:  # Move right
                self.rect.x += self.speed
            elif action == 2:  # Shoot
                self.shoot()
        else:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                self.rect.x -= self.speed
            if keys[pygame.K_RIGHT]:
                self.rect.x += self.speed
            if keys[pygame.K_SPACE]:
                self.shoot()
        self.rect.clamp_ip(screen.get_rect())

    def shoot(self):
        bullet = Bullet(self.rect.centerx, self.rect.top)
        self.game_state.all_sprites.add(bullet)
        self.game_state.bullets.add(bullet)

# Enemy
class Enemy(pygame.sprite.Sprite):
    def __init__(self, x, y, enemy_type):
        super().__init__()
        self.image = pygame.Surface((30, 30))
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.enemy_type = enemy_type
        self.speed = 2
        self.dive_speed = 5
        self.formation_x = x
        self.formation_y = y
        self.state = "formation"
        self.dive_target = None
        self.time = 0

    def update(self, player):
        self.time += 1
        if self.state == "formation":
            self.move_in_formation()
            if self.enemy_type != "stationary" and random.random() < 0.002:  # 0.2% chance to start diving each frame
                self.start_dive(player)
        elif self.state == "diving":
            self.dive()

    def move_in_formation(self):
        # Move in a figure-8 pattern
        t = self.time * 0.05
        offset_x = math.sin(t) * 50
        offset_y = math.sin(t * 2) * 20
        self.rect.x = self.formation_x + offset_x
        self.rect.y = self.formation_y + offset_y

    def start_dive(self, player):
        self.state = "diving"
        self.dive_target = (player.rect.centerx, HEIGHT + 50)  # Aim below the screen

    def dive(self):
        dx = self.dive_target[0] - self.rect.centerx
        dy = self.dive_target[1] - self.rect.centery
        distance = math.hypot(dx, dy)
        if distance > 0:
            dx, dy = dx / distance, dy / distance
            self.rect.x += dx * self.dive_speed
            self.rect.y += dy * self.dive_speed

        # Return to formation if off-screen
        if self.rect.top > HEIGHT:
            self.state = "formation"
            self.rect.center = (self.formation_x, self.formation_y)

# Bullet
class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((5, 10))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.bottom = y
        self.rect.centerx = x
        self.speed = -10

    def update(self):
        self.rect.y += self.speed
        if self.rect.bottom < 0:
            self.kill()

# Game state
class GameState:
    def __init__(self):
        self.player = Player(self)
        self.enemies = pygame.sprite.Group()
        self.bullets = pygame.sprite.Group()
        self.all_sprites = pygame.sprite.Group(self.player)
        self.score = 0
        self.lives = 3
        self.enemies_killed = 0
        self.spawn_enemies()

    def spawn_enemies(self):
        enemy_types = ["diver", "diver", "stationary"]
        rows, cols = 3, 6
        for row in range(rows):
            for col in range(cols):
                x = col * 60 + 100
                y = row * 50 + 50
                enemy_type = random.choice(enemy_types)
                enemy = Enemy(x, y, enemy_type)
                self.enemies.add(enemy)
                self.all_sprites.add(enemy)

    def get_state(self):
        # Update this method to include relevant game state information
        state = [
            self.player.rect.centerx / WIDTH,
            len(self.enemies) / 20,
            self.lives / 3,
            self.enemies_killed / 100,
            *[enemy.rect.centery / HEIGHT for enemy in self.enemies],
            *[enemy.rect.centerx / WIDTH for enemy in self.enemies],
            *[bullet.rect.centery / HEIGHT for bullet in self.bullets],
            *[bullet.rect.centerx / WIDTH for bullet in self.bullets]
        ]
        # Ensure the state has a consistent size
        max_state_size = 100
        state = state[:max_state_size] 
        state += [0] * (max_state_size - len(state))
        return np.array(state)

    def step(self, action):
        self.player.update(action)
        self.enemies.update(self.player)
        self.bullets.update()
        
        reward = 0
        done = False

        # Check for collisions
        for enemy in pygame.sprite.spritecollide(self.player, self.enemies, True):
            self.lives -= 1
            if self.lives <= 0:
                return self.get_state(), -10, True  # Game over
            self.spawn_enemy()

        for enemy in pygame.sprite.groupcollide(self.enemies, self.bullets, True, True):
            self.score += 10
            self.enemies_killed += 1
            reward += 5  # Reward for killing an enemy
            self.spawn_enemy()

        # Small penalty for shooting (to encourage accuracy)
        if action == 2: 
            reward -= 0.1

        if self.lives <= 0:
            done = True

        return self.get_state(), reward, done
    
    def spawn_enemy(self):
            # Method to spawn a single enemy
            enemy_types = ["diver", "diver", "stationary"]
            x = random.randint(50, WIDTH - 50)
            y = random.randint(50, 200)
            enemy_type = random.choice(enemy_types)
            enemy = Enemy(x, y, enemy_type)
            self.enemies.add(enemy)
            self.all_sprites.add(enemy)


def run_game(game_state, ai_player=None, fast_mode=False):
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if ai_player:
            state = game_state.get_state()
            action = ai_player.get_action(state)
        else:
            action = None

        state, reward, done = game_state.step(action)

        if not fast_mode:
            screen.fill(BLACK)
            game_state.all_sprites.draw(screen)
            
            # Draw score and lives
            font = pygame.font.Font(None, 36)
            score_text = font.render(f'Score: {game_state.score}', True, WHITE)
            lives_text = font.render(f'Lives: {game_state.lives}', True, WHITE)
            screen.blit(score_text, (10, 10))
            screen.blit(lives_text, (WIDTH - 110, 10))

            pygame.display.flip()
            clock.tick(60)

        if done:
            running = False

    return game_state.score

# Neural Network
class GalagaNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(GalagaNN, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Genetic Algorithm for NAS
class GeneticNAS:
    def __init__(self, input_size, output_size, population_size=50, mutation_rate=0.1):
        self.input_size = input_size
        self.output_size = output_size
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()

    def initialize_population(self):
        return [self.generate_random_architecture() for _ in range(self.population_size)]

    def generate_random_architecture(self):
        num_layers = random.randint(1, 5)
        hidden_layers = [random.randint(8, 64) for _ in range(num_layers)]
        return GalagaNN(self.input_size, hidden_layers, self.output_size)

    def mutate(self, architecture):
        hidden_layers = list(architecture.network)[:-1:2]  # Extract hidden layers
        if random.random() < self.mutation_rate:
            if random.random() < 0.5 and len(hidden_layers) > 1:
                # Remove a layer
                del hidden_layers[random.randint(0, len(hidden_layers) - 1)]
            else:
                # Add a layer
                hidden_layers.insert(random.randint(0, len(hidden_layers)), nn.Linear(random.randint(8, 64), random.randint(8, 64)))
        
        # Mutate layer sizes
        for layer in hidden_layers:
            if random.random() < self.mutation_rate:
                layer.out_features = random.randint(8, 64)
        
        return self.rebuild_network(hidden_layers)

    def rebuild_network(self, hidden_layers):
        layers = []
        prev_size = self.input_size
        for layer in hidden_layers:
            layers.append(nn.Linear(prev_size, layer.out_features))
            layers.append(nn.ReLU())
            prev_size = layer.out_features
        layers.append(nn.Linear(prev_size, self.output_size))
        return nn.Sequential(*layers)

    def crossover(self, parent1, parent2):
        hidden_layers1 = list(parent1.network)[:-1:2]
        hidden_layers2 = list(parent2.network)[:-1:2]
    
        if len(hidden_layers1) == 1 or len(hidden_layers2) == 1:
            # If either parent has only one layer, return a copy of the longer parent
            return self.rebuild_network(hidden_layers1 if len(hidden_layers1) > len(hidden_layers2) else hidden_layers2)
        
        crossover_point = random.randint(1, min(len(hidden_layers1), len(hidden_layers2)) - 1)
        child_layers = hidden_layers1[:crossover_point] + hidden_layers2[crossover_point:]
        
        return self.rebuild_network(child_layers)

    def evolve(self, fitnesses):
        sorted_population = [x for _, x in sorted(zip(fitnesses, self.population), key=lambda pair: pair[0], reverse=True)]
        
        # Keep top 10% as elites
        num_elites = max(2, self.population_size // 10)
        new_population = sorted_population[:num_elites]
        
        # Fill the rest of the population with offspring
        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(sorted_population[:self.population_size // 2], 2)
            child = GalagaNN(self.input_size, [], self.output_size)
            child.network = self.crossover(parent1, parent2)
            child.network = self.mutate(child)
            new_population.append(child)
        
        self.population = new_population

# AI Player
class AIPlayer:
    def __init__(self, model):
        self.model = model

    def get_action(self, state):
        with torch.no_grad():
            q_values = self.model(torch.FloatTensor(state))
            return q_values.argmax().item()

# Training loop
def train_generation(nas, game_state, iteration, episodes_per_network=10, max_steps=10000):
    fitnesses = []
    for i, network in enumerate(tqdm.tqdm(nas.population, desc="Training Networks")):
        ai_player = AIPlayer(network)
        total_reward = 0
        for episode in range(episodes_per_network):
            game_state = GameState()
            done = False
            steps = 0
            while not done and steps < max_steps:
                state = game_state.get_state()
                action = ai_player.get_action(state)
                _, reward, done = game_state.step(action)
                total_reward += reward
                steps += 1
                
                # Visualize every 100th episode of the first network
                if i == 0 and episode % 100 == 0:
                    visualize_episode(game_state, iteration)
            
            if i == 0 and episode % 100 == 0:
                print(f"Episode {episode} completed. Final round: {iteration}")
        
        fitnesses.append(total_reward / episodes_per_network)
    return fitnesses

def visualize_episode(game_state, iteration):
    screen.fill(BLACK)
    game_state.all_sprites.draw(screen)
    
    # Draw score, lives, and round
    font = pygame.font.Font(None, 36)
    score_text = font.render(f'Score: {game_state.score}', True, WHITE)
    lives_text = font.render(f'Lives: {game_state.lives}', True, WHITE)
    round_text = font.render(f'Round: {iteration}', True, WHITE)
    screen.blit(score_text, (10, 10))
    screen.blit(lives_text, (WIDTH - 110, 10))
    screen.blit(round_text, (WIDTH // 2 - 50, 10))

    pygame.display.flip()
    pygame.time.wait(12)  # Slow down the visualization
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

# Main training loop
def main():
    num_generations = 10
    game_state = GameState()
    input_size = len(game_state.get_state())
    output_size = 3  # Left, Right, Shoot
    iteration = 1
    
    nas = GeneticNAS(input_size, output_size)
    
    for generation in tqdm.tqdm(range(num_generations), desc="Generations"):
        fitnesses = train_generation(nas, game_state, iteration)
        best_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness:.2f}, Avg Fitness = {avg_fitness:.2f}")
        iteration += 1
        nas.evolve(fitnesses)
    
    # Get the best network
    best_network = nas.population[fitnesses.index(max(fitnesses))]
    
    # Test the best network
    ai_player = AIPlayer(best_network)
    game_state.__init__()  # Reset game state
    score = run_game(game_state, ai_player, fast_mode=False)
    print(f"Final Score of Best Network: {score}")

    # Keep the window open until the user closes it
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
    
    pygame.quit()

if __name__ == "__main__":
    main()