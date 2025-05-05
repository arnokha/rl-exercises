import unittest
import numpy as np
from blackjack import BlackjackEnv

class TestBlackjackEnv(unittest.TestCase):
    def setUp(self):
        # Create a fresh environment instance for each test
        self.env = BlackjackEnv()
        # Set random seed for reproducibility
        np.random.seed(42)
        
    def test_init_and_reset(self):
        """Test that initialization and reset work correctly"""
        state = self.env.reset()
        
        # Check that state has the correct format
        self.assertTrue(isinstance(state, tuple))
        self.assertEqual(len(state), 3)
        
        # Unpack state
        player_sum, dealer_card, usable_ace = state
        
        # Check state values are within expected ranges
        self.assertTrue(2 <= player_sum <= 21)
        self.assertTrue(1 <= dealer_card <= 10)
        self.assertIn(usable_ace, [True, False])
        
        # Check that cards were dealt
        self.assertEqual(len(self.env.player_cards), 2)
        self.assertEqual(len(self.env.dealer_cards), 2)
        
    def test_draw_card(self):
        """Test that _draw_card returns values in the correct range"""
        for _ in range(100):  # Draw 100 cards to ensure we get a variety
            card = self.env._draw_card()
            self.assertTrue(1 <= card <= 10)
            
    def test_usable_ace(self):
        """Test the _usable_ace method"""
        # Test cases with a usable ace
        self.assertTrue(self.env._usable_ace([1, 5]))  # Ace + 5 = 16 (with ace as 11)
        self.assertTrue(self.env._usable_ace([1, 10]))  # Ace + 10 = 21 (with ace as 11)
        
        # Test cases without a usable ace
        self.assertFalse(self.env._usable_ace([2, 5]))  # No ace
        self.assertFalse(self.env._usable_ace([1, 10, 5]))  # Ace + 10 + 5 = 16 (ace must be 1)
        
    def test_step_hit(self):
        """Test the step method when action is hit (1)"""
        self.env.reset()
        
        # Record initial state
        initial_cards_count = len(self.env.player_cards)
        
        # Take hit action
        next_state, reward, done, _ = self.env.step(1)  # Hit
        
        # Check that player received a new card
        self.assertEqual(len(self.env.player_cards), initial_cards_count + 1)
        
        # Check that next_state reflects the new card
        player_sum, dealer_card, usable_ace = next_state
        self.assertEqual(dealer_card, self.env.dealer_cards[0])
        
        # If player busts, done should be True and reward should be -1
        if player_sum > 21:
            self.assertTrue(done)
            self.assertEqual(reward, -1)
            
    def test_step_stick(self):
        """Test the step method when action is stick (0)"""
        self.env.reset()
        
        # If we stick, the episode should end
        next_state, reward, done, _ = self.env.step(0)  # Stick
        
        # Check that episode is done
        self.assertTrue(done)
        
        # Check that dealer played their turn (might have drawn more cards)
        self.assertTrue(len(self.env.dealer_cards) >= 2)
        
        # Reward should be -1, 0, or 1
        self.assertIn(reward, [-1, 0, 1])
        
    def test_dealer_plays(self):
        """Test that dealer follows the rule of hitting until sum is at least 17"""
        # Set up an environment with a specific scenario
        self.env.reset()
        
        # Force dealer to have a low sum by setting dealer cards to [2, 3]
        self.env.dealer_cards = [2, 3]
        
        # Player sticks
        self.env.step(0)
        
        # Check that dealer drew more cards
        self.assertTrue(len(self.env.dealer_cards) > 2)
        
        # Calculate final dealer sum
        dealer_sum = sum(self.env.dealer_cards)
        if 1 in self.env.dealer_cards and dealer_sum + 10 <= 21:
            dealer_sum += 10
            
        # Dealer either has sum >= 17 or busted
        self.assertTrue(dealer_sum >= 17 or dealer_sum > 21)
        
    def test_game_outcomes(self):
        """Test various game outcomes"""
        # Test player bust
        self.env.reset()
        self.env.player_cards = [10, 10, 5]  # Sum = 25, player busts
        _, reward, done, _ = self.env.step(1)  # Hit and bust
        self.assertTrue(done)
        self.assertEqual(reward, -1)
        
        # Test dealer bust
        self.env.reset()
        self.env.player_cards = [10, 8]  # Sum = 18
        self.env.dealer_cards = [10, 6, 9]  # Sum = 25, dealer busts
        _, reward, done, _ = self.env.step(0)  # Stick
        self.assertTrue(done)
        self.assertEqual(reward, 1)  # Player wins
        
        # Test player wins
        self.env.reset()
        self.env.player_cards = [10, 9]  # Sum = 19
        self.env.dealer_cards = [10, 7]  # Sum = 17
        _, reward, done, _ = self.env.step(0)  # Stick
        self.assertTrue(done)
        self.assertEqual(reward, 1)  # Player wins
        
        # Test dealer wins
        self.env.reset()
        self.env.player_cards = [10, 7]  # Sum = 17
        self.env.dealer_cards = [10, 9]  # Sum = 19
        _, reward, done, _ = self.env.step(0)  # Stick
        self.assertTrue(done)
        self.assertEqual(reward, -1)  # Dealer wins
        
        # Test tie
        self.env.reset()
        self.env.player_cards = [10, 9]  # Sum = 19
        self.env.dealer_cards = [10, 9]  # Sum = 19
        _, reward, done, _ = self.env.step(0)  # Stick
        self.assertTrue(done)
        self.assertEqual(reward, 0)  # Tie
        
    def test_invalid_action(self):
        """Test that invalid actions raise an error"""
        self.env.reset()
        
        # Action should be 0 or 1
        with self.assertRaises(AssertionError):
            self.env.step(2)  # Invalid action
            
    def test_step_after_done(self):
        """Test that calling step after the episode is done raises an error"""
        self.env.reset()
        
        # End the episode
        self.env.step(0)  # Stick
        
        # Calling step again should raise an error
        with self.assertRaises(AssertionError):
            self.env.step(0)

if __name__ == "__main__":
    unittest.main()