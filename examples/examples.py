#!/usr/bin/env python3
"""
Example scenarios for TravelingFlame ring simulation.
"""

import sys
import os
import math

# Add parent directory to path to import flame_ring
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flame_ring import RingTrough, FlameProperties


def demo_dual_ignition():
    """Demonstrate flames propagating from two points meeting in the middle."""
    print("Demo: Dual Ignition - Flames from opposite sides")
    print("=" * 50)
    
    ring = RingTrough(num_segments=20)
    
    # Set uniform fuel
    for segment in ring.segments:
        segment.fuel_level = 0.9
    
    # Ignite two opposite segments
    ring.ignite_segment(0)
    ring.ignite_segment(10)
    
    print("Initial state - two ignition points:")
    print(ring.visualize_ring())
    print(ring.get_state_summary())
    print()
    
    # Run simulation
    step = 0
    while ring.time_elapsed < 8.0 and ring.is_any_burning():
        ring.update(0.1)
        step += 1
        
        if step % 15 == 0:
            print(f"Step {step} (t={ring.time_elapsed:.1f}s):")
            print(ring.visualize_ring())
            print(ring.get_state_summary())
            print()


def demo_fuel_gradient():
    """Demonstrate propagation through varying fuel density."""
    print("Demo: Fuel Gradient - Non-uniform fuel distribution")
    print("=" * 50)
    
    ring = RingTrough(num_segments=16)
    
    # Create fuel gradient - more fuel on one side
    for i, segment in enumerate(ring.segments):
        # Sinusoidal fuel distribution
        fuel_factor = 0.3 + 0.7 * (1 + math.cos(2 * math.pi * i / len(ring.segments))) / 2
        segment.fuel_level = fuel_factor
    
    # Ignite at low fuel area
    ring.ignite_segment(8)
    
    print("Initial state - gradient fuel (more fuel = more 'o'):")
    print(ring.visualize_ring())
    print(ring.get_state_summary())
    print()
    
    # Run simulation
    step = 0
    while ring.time_elapsed < 12.0 and ring.is_any_burning():
        ring.update(0.1)
        step += 1
        
        if step % 20 == 0:
            print(f"Step {step} (t={ring.time_elapsed:.1f}s):")
            print(ring.visualize_ring())
            print(ring.get_state_summary())
            print()


def demo_slow_burn():
    """Demonstrate a slow-burning scenario with less fuel consumption."""
    print("Demo: Slow Burn - Low consumption rate")
    print("=" * 50)
    
    ring = RingTrough(num_segments=12)
    
    # Modify flame properties for slower burn
    ring.flame_properties.fuel_consumption_rate = 0.03  # Much slower
    ring.flame_properties.propagation_speed = 0.5
    
    # Set fuel levels
    for segment in ring.segments:
        segment.fuel_level = 0.6
    
    ring.ignite_segment(0)
    
    print("Initial state - slow burn simulation:")
    print(ring.visualize_ring())
    print(ring.get_state_summary())
    print()
    
    # Run longer simulation
    step = 0
    while ring.time_elapsed < 15.0 and ring.is_any_burning():
        ring.update(0.1)
        step += 1
        
        if step % 30 == 0:
            print(f"Step {step} (t={ring.time_elapsed:.1f}s):")
            print(ring.visualize_ring())
            print(ring.get_state_summary())
            print()


def demo_large_ring():
    """Demonstrate propagation in a larger ring."""
    print("Demo: Large Ring - 32 segments")
    print("=" * 50)
    
    ring = RingTrough(num_segments=32)
    
    # Set fuel with some variation
    for i, segment in enumerate(ring.segments):
        base_fuel = 0.7
        variation = 0.2 * math.sin(i * 0.3)
        segment.fuel_level = max(0.1, base_fuel + variation)
    
    ring.ignite_segment(0)
    
    print("Initial state - large ring:")
    print(ring.visualize_ring())
    print(ring.get_state_summary())
    print()
    
    # Run simulation with less frequent updates for readability
    step = 0
    while ring.time_elapsed < 10.0 and ring.is_any_burning():
        ring.update(0.1)
        step += 1
        
        if step % 25 == 0:
            print(f"Step {step} (t={ring.time_elapsed:.1f}s):")
            print(ring.visualize_ring())
            print(ring.get_state_summary())
            print()


if __name__ == "__main__":
    demos = {
        "1": ("Dual Ignition", demo_dual_ignition),
        "2": ("Fuel Gradient", demo_fuel_gradient),
        "3": ("Slow Burn", demo_slow_burn),
        "4": ("Large Ring", demo_large_ring),
    }
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice in demos:
            demos[choice][1]()
        else:
            print("Invalid choice. Available demos:")
            for key, (name, _) in demos.items():
                print(f"  {key}: {name}")
    else:
        print("TravelingFlame Demo Scenarios")
        print("=" * 30)
        print("Available demos:")
        for key, (name, _) in demos.items():
            print(f"  {key}: {name}")
        print()
        print("Usage: python examples.py [demo_number]")
        print("   or: python examples.py [demo_number]")
        print()
        
        # Run all demos
        for key, (name, func) in demos.items():
            print()
            func()
            print("\n" + "="*60 + "\n")