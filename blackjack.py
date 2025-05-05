import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

class BlackjackEnv:
    """
    Blackjack environment following the OpenAI Gym interface.
    
    Actions:
    - 0: Stick (stop receiving cards)
    - 1: Hit (receive another card)
    
    State:
    - Player's current sum
    - Dealer's visible card (first card)
    - Whether player has a usable ace (an ace valued at 11 rather than 1)
    
    Rewards:
    - Win: +1
    - Draw: 0
    - Lose: -1
    """
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.dealer_cards = self._draw_hand()
        self.player_cards = self._draw_hand()
        self.done = False
        return self._get_state()
    
    def _draw_card(self):
        card = min(10, np.random.randint(1, 14))  # Jack, Queen, King = 10
        return card
    
    def _draw_hand(self):
        return [self._draw_card(), self._draw_card()]
    
    def _usable_ace(self, cards):
        # Check if the hand has a usable ace (can be counted as 11 without busting)
        return 1 in cards and sum(cards) + 10 <= 21
    
    def _get_state(self):
        # Calculate the player's current sum
        player_sum = sum(self.player_cards)
        if self._usable_ace(self.player_cards):
            player_sum += 10
            
        # Get the dealer's visible card (first card)
        dealer_card = self.dealer_cards[0]
        
        # Check if player has a usable ace
        usable_ace = self._usable_ace(self.player_cards)
        
        return (player_sum, dealer_card, usable_ace)
    
    def step(self, action):
        assert not self.done, "Episode already done"
        assert action in [0, 1], f"Invalid action: {action}"
        
        reward = 0
        
        # Player's turn
        if action == 1:  # Hit: player gets a new card
            self.player_cards.append(self._draw_card())
            
            # Calculate player sum considering aces
            player_sum = sum(self.player_cards)
            if self._usable_ace(self.player_cards):
                player_sum += 10
                
            # Check if player busts
            if player_sum > 21:
                self.done = True
                reward = -1
        
        else:  # action == 0 (Stick): dealer's turn
            self.done = True
            
            # Calculate player sum considering aces
            player_sum = sum(self.player_cards)
            if self._usable_ace(self.player_cards):
                player_sum += 10
            
            # Dealer hits until sum is at least 17
            dealer_sum = sum(self.dealer_cards)
            if self._usable_ace(self.dealer_cards):
                dealer_sum += 10
                
            while dealer_sum < 17:
                self.dealer_cards.append(self._draw_card())
                dealer_sum = sum(self.dealer_cards)
                if self._usable_ace(self.dealer_cards):
                    dealer_sum += 10
                if dealer_sum > 21:  # Dealer busts
                    break
            
            # Determine reward
            if dealer_sum > 21:  # Dealer busts
                reward = 1
            elif dealer_sum > player_sum:  # Dealer wins
                reward = -1
            elif dealer_sum < player_sum:  # Player wins
                reward = 1
            else:  # Tie
                reward = 0
                
        return self._get_state(), reward, self.done, {}


def monte_carlo_policy_evaluation(env, policy, num_episodes=10000):
    """
    Monte Carlo policy evaluation to estimate state values under a given policy.
    
    Args:
        env: The Blackjack environment
        policy: Function that takes a state and returns an action
        num_episodes: Number of episodes to sample
    
    Returns:
        V: State-value function as a 3D numpy array (player_sum, dealer_card, usable_ace)
        counts: Count of visits to each state
    """
    # Initialize value function and counts
    V = np.zeros((22, 11, 2))  # (player_sum, dealer_card, usable_ace)
    counts = np.zeros((22, 11, 2))
    
    # Simulate episodes
    for _ in range(num_episodes):
        # Generate an episode
        episode = []
        state = env.reset()
        done = False
        
        while not done:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
        
        # Extract states and returns from episode
        states = [s for s, _, _ in episode]
        rewards = [r for _, _, r in episode]
        
        # Update value function using first-visit MC
        G = 0
        for t in range(len(episode)-1, -1, -1):
            G += rewards[t]
            
            state = states[t]
            player_sum, dealer_card, usable_ace = state
            
            # Skip if this state was already visited in this episode
            if state in states[:t]:
                continue
                
            # Update value function estimate
            counts[player_sum, dealer_card, int(usable_ace)] += 1
            V[player_sum, dealer_card, int(usable_ace)] += (G - V[player_sum, dealer_card, int(usable_ace)]) / counts[player_sum, dealer_card, int(usable_ace)]
    
    return V, counts


def plot_value_function(V, title="Value Function", filename=None):
    """
    Plot the value function as a surface.
    
    Args:
        V: Value function as a 3D numpy array (player_sum, dealer_card, usable_ace)
        title: Title for the plot
        filename: If provided, save the plot to this file instead of displaying it
    """
    fig = plt.figure(figsize=(16, 8))
    
    # Plot with usable ace
    ax1 = fig.add_subplot(121, projection='3d')
    x_range = np.arange(1, 11)
    y_range = np.arange(12, 22)
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.array([V[y, x, 1] for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)
    
    surf1 = ax1.plot_surface(X, Y, Z, cmap='viridis')
    plt.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
    ax1.set_xlabel('Dealer showing')
    ax1.set_ylabel('Player sum')
    ax1.set_zlabel('Value')
    ax1.set_title(f'{title} (Usable Ace)')
    
    # Plot without usable ace
    ax2 = fig.add_subplot(122, projection='3d')
    Z = np.array([V[y, x, 0] for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)
    
    surf2 = ax2.plot_surface(X, Y, Z, cmap='viridis')
    plt.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
    ax2.set_xlabel('Dealer showing')
    ax2.set_ylabel('Player sum')
    ax2.set_zlabel('Value')
    ax2.set_title(f'{title} (No Usable Ace)')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Value function plot saved to: {filename}")
    else:
        plt.show()


def main():
    # Create Blackjack environment
    env = BlackjackEnv()
    
    # Define a simple policy (always hit if sum < 20, otherwise stick)
    def simple_policy(state):
        player_sum, _, _ = state
        return 1 if player_sum < 20 else 0
    
    # Evaluate the policy
    V, counts = monte_carlo_policy_evaluation(env, simple_policy, num_episodes=5000)
    
    # Plot the value function and save to file
    plot_value_function(V, title="Simple Policy Value Function", filename="blackjack_value_function.png")
    
    print("Policy evaluation complete. The value function has been plotted and saved.")


if __name__ == "__main__":
    main()