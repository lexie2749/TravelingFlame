#!/usr/bin/env python3
"""
Configuration-based runner for TravelingFlame simulations.
Allows loading simulation parameters from JSON files.
"""

import json
import sys
import os
from flame_ring import RingTrough, FlameProperties, run_simulation


def load_config(config_file: str) -> dict:
    """Load configuration from JSON file."""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{config_file}': {e}")
        sys.exit(1)


def create_scenario_from_config(config: dict) -> RingTrough:
    """Create a ring trough scenario from configuration."""
    # Ring parameters
    ring_config = config.get('ring', {})
    num_segments = ring_config.get('num_segments', 16)
    radius = ring_config.get('radius', 1.0)
    
    ring = RingTrough(num_segments=num_segments, radius=radius)
    
    # Flame properties
    flame_config = config.get('flame_properties', {})
    ring.flame_properties = FlameProperties(
        temperature=flame_config.get('temperature', 800.0),
        propagation_speed=flame_config.get('propagation_speed', 1.0),
        fuel_consumption_rate=flame_config.get('fuel_consumption_rate', 0.1),
        ignition_temperature=flame_config.get('ignition_temperature', 200.0)
    )
    
    # Fuel distribution
    fuel_config = config.get('fuel_distribution', {})
    fuel_type = fuel_config.get('type', 'uniform')
    
    if fuel_type == 'uniform':
        fuel_level = fuel_config.get('level', 0.8)
        for segment in ring.segments:
            segment.fuel_level = fuel_level
    
    elif fuel_type == 'gradient':
        import math
        base_level = fuel_config.get('base_level', 0.5)
        amplitude = fuel_config.get('amplitude', 0.3)
        frequency = fuel_config.get('frequency', 1.0)
        
        for i, segment in enumerate(ring.segments):
            angle = 2 * math.pi * i / num_segments
            variation = amplitude * math.sin(frequency * angle)
            segment.fuel_level = max(0.0, base_level + variation)
    
    elif fuel_type == 'custom':
        levels = fuel_config.get('levels', [])
        for i, segment in enumerate(ring.segments):
            if i < len(levels):
                segment.fuel_level = levels[i]
            else:
                segment.fuel_level = 0.5  # Default
    
    # Initial ignition points
    ignition_config = config.get('ignition', {})
    ignition_points = ignition_config.get('points', [0])
    
    for point in ignition_points:
        if 0 <= point < num_segments:
            ring.ignite_segment(point)
    
    return ring


def run_config_simulation(config_file: str):
    """Run simulation from configuration file."""
    config = load_config(config_file)
    
    # Simulation parameters
    sim_config = config.get('simulation', {})
    duration = sim_config.get('duration', 10.0)
    dt = sim_config.get('time_step', 0.1)
    show_visualization = sim_config.get('show_visualization', True)
    update_interval = sim_config.get('update_interval', 10)
    
    # Create scenario
    ring = create_scenario_from_config(config)
    
    # Print scenario info
    scenario_name = config.get('name', 'Custom Scenario')
    description = config.get('description', 'No description provided')
    
    print(f"TravelingFlame: {scenario_name}")
    print("=" * 50)
    print(f"Description: {description}")
    print(f"Segments: {ring.num_segments}")
    print(f"Duration: {duration}s")
    print(f"Initial ignition points: {config.get('ignition', {}).get('points', [0])}")
    print()
    
    if show_visualization:
        print("Legend: █▬▪· = flame intensity, o = fuel, . = low fuel, space = empty")
        print()
        print("Initial state:")
        print(ring.visualize_ring())
        print(ring.get_state_summary())
        print()
    
    # Run simulation
    step = 0
    while ring.time_elapsed < duration and ring.is_any_burning():
        ring.update(dt)
        step += 1
        
        # Show periodic updates
        if show_visualization and step % update_interval == 0:
            print(f"Step {step}:")
            print(ring.visualize_ring())
            print(ring.get_state_summary())
            print()
        elif step % (update_interval * 5) == 0:
            print(ring.get_state_summary())
    
    # Final state
    print("Final state:")
    if show_visualization:
        print(ring.visualize_ring())
    print(ring.get_state_summary())
    
    if not ring.is_any_burning():
        print("Flame has extinguished.")
    else:
        print("Simulation completed with flame still burning.")


def create_example_configs():
    """Create example configuration files."""
    os.makedirs('configs', exist_ok=True)
    
    # Basic scenario
    basic_config = {
        "name": "Basic Ring Propagation",
        "description": "Simple flame propagation around a uniform ring",
        "ring": {
            "num_segments": 16,
            "radius": 1.0
        },
        "flame_properties": {
            "temperature": 800.0,
            "propagation_speed": 1.0,
            "fuel_consumption_rate": 0.1,
            "ignition_temperature": 200.0
        },
        "fuel_distribution": {
            "type": "uniform",
            "level": 0.8
        },
        "ignition": {
            "points": [0]
        },
        "simulation": {
            "duration": 10.0,
            "time_step": 0.1,
            "show_visualization": True,
            "update_interval": 10
        }
    }
    
    # Dual ignition scenario
    dual_config = {
        "name": "Dual Ignition Meeting",
        "description": "Two flames start from opposite sides and meet in the middle",
        "ring": {
            "num_segments": 20,
            "radius": 1.5
        },
        "flame_properties": {
            "temperature": 900.0,
            "propagation_speed": 1.2,
            "fuel_consumption_rate": 0.08,
            "ignition_temperature": 180.0
        },
        "fuel_distribution": {
            "type": "uniform",
            "level": 0.9
        },
        "ignition": {
            "points": [0, 10]
        },
        "simulation": {
            "duration": 8.0,
            "time_step": 0.1,
            "show_visualization": True,
            "update_interval": 15
        }
    }
    
    # Gradient scenario
    gradient_config = {
        "name": "Fuel Gradient Challenge",
        "description": "Varying fuel density creates interesting propagation patterns",
        "ring": {
            "num_segments": 24,
            "radius": 2.0
        },
        "flame_properties": {
            "temperature": 750.0,
            "propagation_speed": 0.8,
            "fuel_consumption_rate": 0.12,
            "ignition_temperature": 220.0
        },
        "fuel_distribution": {
            "type": "gradient",
            "base_level": 0.5,
            "amplitude": 0.4,
            "frequency": 2.0
        },
        "ignition": {
            "points": [6]
        },
        "simulation": {
            "duration": 15.0,
            "time_step": 0.1,
            "show_visualization": True,
            "update_interval": 20
        }
    }
    
    # Save configs
    configs = [
        ('basic.json', basic_config),
        ('dual_ignition.json', dual_config),
        ('fuel_gradient.json', gradient_config)
    ]
    
    for filename, config in configs:
        with open(f'configs/{filename}', 'w') as f:
            json.dump(config, f, indent=2)
    
    print("Created example configuration files in 'configs/' directory:")
    for filename, _ in configs:
        print(f"  - configs/{filename}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("TravelingFlame Configuration Runner")
        print("=" * 40)
        print("Usage:")
        print("  python config_runner.py <config_file.json>")
        print("  python config_runner.py --create-examples")
        print()
        print("Examples:")
        print("  python config_runner.py configs/basic.json")
        print("  python config_runner.py --create-examples")
        sys.exit(1)
    
    if sys.argv[1] == "--create-examples":
        create_example_configs()
    else:
        config_file = sys.argv[1]
        run_config_simulation(config_file)