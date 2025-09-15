#!/usr/bin/env python3
"""
TravelingFlame: A simulation of flame propagation through a ring-shaped trough.

This module implements a physical simulation where flames can propagate around
a circular trough, with proper boundary conditions for ring topology.
"""

import math
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FlameProperties:
    """Properties that affect flame propagation."""
    temperature: float = 800.0  # Temperature in Celsius
    propagation_speed: float = 1.0  # Speed in units/second
    fuel_consumption_rate: float = 0.1  # Fuel consumed per second
    ignition_temperature: float = 200.0  # Minimum temperature to ignite


class TroughSegment:
    """Represents a segment of the ring-shaped trough."""
    
    def __init__(self, position: int, fuel_level: float = 1.0, temperature: float = 20.0):
        self.position = position
        self.fuel_level = fuel_level  # 0.0 to 1.0
        self.temperature = temperature  # Celsius
        self.is_burning = False
        self.flame_intensity = 0.0  # 0.0 to 1.0
    
    def can_ignite(self, ignition_temp: float) -> bool:
        """Check if this segment can ignite given the ignition temperature."""
        return self.fuel_level > 0.1 and self.temperature >= ignition_temp
    
    def ignite(self, flame_properties: FlameProperties):
        """Ignite this segment if conditions are met."""
        if self.can_ignite(flame_properties.ignition_temperature):
            self.is_burning = True
            self.flame_intensity = min(1.0, self.fuel_level)
            self.temperature = flame_properties.temperature
            return True
        return False
    
    def update(self, dt: float, flame_properties: FlameProperties):
        """Update segment state over time."""
        if self.is_burning:
            # Consume fuel
            fuel_consumed = flame_properties.fuel_consumption_rate * dt
            self.fuel_level = max(0.0, self.fuel_level - fuel_consumed)
            
            # Adjust flame intensity based on remaining fuel
            self.flame_intensity = min(1.0, self.fuel_level * 2.0)
            
            # Extinguish if no fuel remains
            if self.fuel_level <= 0.0:
                self.is_burning = False
                self.flame_intensity = 0.0
                self.temperature = 20.0  # Cool down
    
    def get_heat_output(self) -> float:
        """Get the heat output of this segment."""
        return self.flame_intensity * self.temperature if self.is_burning else 0.0


class RingTrough:
    """A ring-shaped trough where flames can propagate."""
    
    def __init__(self, num_segments: int = 20, radius: float = 1.0):
        self.num_segments = num_segments
        self.radius = radius
        self.segments: List[TroughSegment] = []
        self.flame_properties = FlameProperties()
        self.time_elapsed = 0.0
        
        # Initialize segments
        for i in range(num_segments):
            self.segments.append(TroughSegment(i))
    
    def get_neighbor_indices(self, index: int) -> Tuple[int, int]:
        """Get the indices of neighboring segments (handles ring topology)."""
        left = (index - 1) % self.num_segments
        right = (index + 1) % self.num_segments
        return left, right
    
    def ignite_segment(self, index: int):
        """Manually ignite a specific segment."""
        if 0 <= index < self.num_segments:
            # Force ignition by setting temperature high enough
            self.segments[index].temperature = self.flame_properties.ignition_temperature + 100
            self.segments[index].ignite(self.flame_properties)
    
    def propagate_heat(self, dt: float):
        """Propagate heat between neighboring segments."""
        heat_transfer_rate = 0.5  # Heat transfer coefficient
        
        # Calculate heat transfer for each segment
        heat_changes = [0.0] * self.num_segments
        
        for i, segment in enumerate(self.segments):
            if segment.is_burning:
                left_idx, right_idx = self.get_neighbor_indices(i)
                left_segment = self.segments[left_idx]
                right_segment = self.segments[right_idx]
                
                # Transfer heat to neighbors
                heat_output = segment.get_heat_output()
                heat_per_neighbor = heat_output * heat_transfer_rate * dt
                
                # Heat up neighboring segments
                if not left_segment.is_burning:
                    heat_changes[left_idx] += heat_per_neighbor
                if not right_segment.is_burning:
                    heat_changes[right_idx] += heat_per_neighbor
        
        # Apply heat changes
        for i, heat_change in enumerate(heat_changes):
            if heat_change > 0:
                self.segments[i].temperature += heat_change
                # Try to ignite if hot enough
                if not self.segments[i].is_burning:
                    self.segments[i].ignite(self.flame_properties)
    
    def update(self, dt: float):
        """Update the entire ring simulation."""
        self.time_elapsed += dt
        
        # Update each segment
        for segment in self.segments:
            segment.update(dt, self.flame_properties)
        
        # Propagate heat between segments
        self.propagate_heat(dt)
    
    def get_burning_segments(self) -> List[int]:
        """Get indices of all currently burning segments."""
        return [i for i, segment in enumerate(self.segments) if segment.is_burning]
    
    def is_any_burning(self) -> bool:
        """Check if any segment is currently burning."""
        return any(segment.is_burning for segment in self.segments)
    
    def get_state_summary(self) -> str:
        """Get a text summary of the current state."""
        burning = self.get_burning_segments()
        total_fuel = sum(seg.fuel_level for seg in self.segments)
        avg_temp = sum(seg.temperature for seg in self.segments) / len(self.segments)
        
        return (f"Time: {self.time_elapsed:.1f}s | "
                f"Burning segments: {len(burning)} | "
                f"Total fuel: {total_fuel:.1f} | "
                f"Avg temp: {avg_temp:.1f}°C")
    
    def visualize_ring(self) -> str:
        """Create a simple ASCII visualization of the ring."""
        symbols = []
        for segment in self.segments:
            if segment.is_burning:
                intensity = int(segment.flame_intensity * 3)
                symbol = ['·', '▪', '▬', '█'][intensity]
            elif segment.fuel_level > 0.5:
                symbol = 'o'
            elif segment.fuel_level > 0:
                symbol = '.'
            else:
                symbol = ' '
            symbols.append(symbol)
        
        # Arrange in a rough circle pattern
        n = len(symbols)
        radius = max(3, n // 6)
        grid_size = radius * 2 + 1
        grid = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]
        center = radius
        
        for i, symbol in enumerate(symbols):
            angle = 2 * math.pi * i / n
            x = int(center + radius * math.cos(angle))
            y = int(center + radius * math.sin(angle))
            if 0 <= x < grid_size and 0 <= y < grid_size:
                grid[y][x] = symbol
        
        return '\n'.join(''.join(row) for row in grid)


def run_simulation(duration: float = 10.0, dt: float = 0.1, visualization: bool = True):
    """Run a flame propagation simulation."""
    print("TravelingFlame: Ring-shaped Trough Simulation")
    print("=" * 50)
    print("Legend: █▬▪· = flame intensity, o = fuel, . = low fuel, space = empty")
    print()
    
    # Create ring trough
    ring = RingTrough(num_segments=16)
    
    # Set initial conditions - add fuel to all segments
    for segment in ring.segments:
        segment.fuel_level = 0.8 + 0.2 * math.sin(segment.position * 0.5)  # Varying fuel
    
    # Ignite one segment to start
    ring.ignite_segment(0)
    
    print("Initial state:")
    if visualization:
        print(ring.visualize_ring())
    print(ring.get_state_summary())
    print()
    
    # Run simulation
    step = 0
    while ring.time_elapsed < duration and ring.is_any_burning():
        ring.update(dt)
        step += 1
        
        # Show periodic updates
        if visualization and step % 10 == 0:
            print(f"Step {step}:")
            print(ring.visualize_ring())
            print(ring.get_state_summary())
            print()
        elif step % 50 == 0:
            print(ring.get_state_summary())
    
    print("Final state:")
    if visualization:
        print(ring.visualize_ring())
    print(ring.get_state_summary())
    
    if not ring.is_any_burning():
        print("Flame has extinguished.")
    else:
        print("Simulation completed with flame still burning.")


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    duration = 20.0
    show_viz = True
    
    if len(sys.argv) > 1:
        try:
            duration = float(sys.argv[1])
        except ValueError:
            print("Usage: python flame_ring.py [duration] [--no-viz]")
            sys.exit(1)
    
    if "--no-viz" in sys.argv:
        show_viz = False
    
    run_simulation(duration=duration, visualization=show_viz)