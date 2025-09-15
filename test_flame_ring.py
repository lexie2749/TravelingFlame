#!/usr/bin/env python3
"""
Basic tests for TravelingFlame ring simulation.
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flame_ring import RingTrough, TroughSegment, FlameProperties


class TestTroughSegment(unittest.TestCase):
    """Test individual trough segment functionality."""
    
    def setUp(self):
        self.segment = TroughSegment(position=0, fuel_level=0.8, temperature=20.0)
        self.flame_props = FlameProperties()
    
    def test_initial_state(self):
        """Test initial segment state."""
        self.assertEqual(self.segment.position, 0)
        self.assertEqual(self.segment.fuel_level, 0.8)
        self.assertEqual(self.segment.temperature, 20.0)
        self.assertFalse(self.segment.is_burning)
        self.assertEqual(self.segment.flame_intensity, 0.0)
    
    def test_can_ignite(self):
        """Test ignition conditions."""
        # Should not ignite at room temperature
        self.assertFalse(self.segment.can_ignite(200.0))
        
        # Should ignite at high temperature with fuel
        self.segment.temperature = 250.0
        self.assertTrue(self.segment.can_ignite(200.0))
        
        # Should not ignite without fuel
        self.segment.fuel_level = 0.05
        self.assertFalse(self.segment.can_ignite(200.0))
    
    def test_ignition(self):
        """Test ignition process."""
        self.segment.temperature = 250.0
        result = self.segment.ignite(self.flame_props)
        
        self.assertTrue(result)
        self.assertTrue(self.segment.is_burning)
        self.assertEqual(self.segment.temperature, self.flame_props.temperature)
        self.assertGreater(self.segment.flame_intensity, 0)
    
    def test_fuel_consumption(self):
        """Test fuel consumption during burning."""
        self.segment.temperature = 250.0
        self.segment.ignite(self.flame_props)
        
        initial_fuel = self.segment.fuel_level
        self.segment.update(1.0, self.flame_props)  # 1 second
        
        self.assertLess(self.segment.fuel_level, initial_fuel)
        
        # Continue until fuel runs out
        while self.segment.fuel_level > 0:
            self.segment.update(1.0, self.flame_props)
        
        self.assertFalse(self.segment.is_burning)
        self.assertEqual(self.segment.flame_intensity, 0.0)
    
    def test_heat_output(self):
        """Test heat output calculation."""
        # No heat when not burning
        self.assertEqual(self.segment.get_heat_output(), 0.0)
        
        # Heat when burning
        self.segment.temperature = 250.0
        self.segment.ignite(self.flame_props)
        self.assertGreater(self.segment.get_heat_output(), 0.0)


class TestRingTrough(unittest.TestCase):
    """Test ring trough functionality."""
    
    def setUp(self):
        self.ring = RingTrough(num_segments=8)
    
    def test_ring_creation(self):
        """Test ring initialization."""
        self.assertEqual(len(self.ring.segments), 8)
        self.assertEqual(self.ring.num_segments, 8)
        self.assertEqual(self.ring.time_elapsed, 0.0)
    
    def test_neighbor_indices(self):
        """Test neighbor calculation with ring topology."""
        # Test middle segment
        left, right = self.ring.get_neighbor_indices(3)
        self.assertEqual(left, 2)
        self.assertEqual(right, 4)
        
        # Test first segment (wraps to last)
        left, right = self.ring.get_neighbor_indices(0)
        self.assertEqual(left, 7)  # Wraps around
        self.assertEqual(right, 1)
        
        # Test last segment (wraps to first)
        left, right = self.ring.get_neighbor_indices(7)
        self.assertEqual(left, 6)
        self.assertEqual(right, 0)  # Wraps around
    
    def test_manual_ignition(self):
        """Test manual segment ignition."""
        # Add fuel to all segments
        for segment in self.ring.segments:
            segment.fuel_level = 0.8
        
        self.ring.ignite_segment(0)
        self.assertTrue(self.ring.segments[0].is_burning)
        self.assertEqual(len(self.ring.get_burning_segments()), 1)
    
    def test_propagation(self):
        """Test flame propagation between segments."""
        # Set up fuel
        for segment in self.ring.segments:
            segment.fuel_level = 0.8
        
        # Ignite one segment
        self.ring.ignite_segment(0)
        initial_burning = len(self.ring.get_burning_segments())
        
        # Run simulation for a few steps
        for _ in range(20):
            self.ring.update(0.1)
        
        # Should have more burning segments due to propagation
        final_burning = len(self.ring.get_burning_segments())
        self.assertGreater(final_burning, initial_burning)
    
    def test_fuel_depletion(self):
        """Test that flames eventually go out when fuel is depleted."""
        # Set up limited fuel
        for segment in self.ring.segments:
            segment.fuel_level = 0.2  # Low fuel
        
        self.ring.ignite_segment(0)
        
        # Run until no flames remain
        max_iterations = 1000
        iterations = 0
        while self.ring.is_any_burning() and iterations < max_iterations:
            self.ring.update(0.1)
            iterations += 1
        
        # Should eventually burn out
        self.assertFalse(self.ring.is_any_burning())
        self.assertLess(iterations, max_iterations)  # Shouldn't take forever
    
    def test_ring_topology_propagation(self):
        """Test that flames can propagate around the full ring."""
        # Set up ring with fuel
        for segment in self.ring.segments:
            segment.fuel_level = 0.6
        
        # Ignite first segment
        self.ring.ignite_segment(0)
        
        # Run simulation
        for _ in range(100):
            self.ring.update(0.1)
            if len(self.ring.get_burning_segments()) >= self.ring.num_segments:
                break
        
        # Should eventually ignite all segments
        self.assertEqual(len(self.ring.get_burning_segments()), self.ring.num_segments)


class TestFlameProperties(unittest.TestCase):
    """Test flame properties."""
    
    def test_default_properties(self):
        """Test default flame properties."""
        props = FlameProperties()
        self.assertEqual(props.temperature, 800.0)
        self.assertEqual(props.propagation_speed, 1.0)
        self.assertEqual(props.fuel_consumption_rate, 0.1)
        self.assertEqual(props.ignition_temperature, 200.0)
    
    def test_custom_properties(self):
        """Test custom flame properties."""
        props = FlameProperties(
            temperature=1000.0,
            propagation_speed=2.0,
            fuel_consumption_rate=0.2,
            ignition_temperature=300.0
        )
        self.assertEqual(props.temperature, 1000.0)
        self.assertEqual(props.propagation_speed, 2.0)
        self.assertEqual(props.fuel_consumption_rate, 0.2)
        self.assertEqual(props.ignition_temperature, 300.0)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)