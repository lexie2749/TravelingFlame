# TravelingFlame

A simulation of flame propagation through a ring-shaped trough.

## Overview

This project implements a physical simulation where flames can propagate around a circular trough, with proper boundary conditions for ring topology. The simulation models:

- Ring-shaped geometry with periodic boundary conditions
- Flame propagation based on physical properties (fuel level, temperature, oxygen)
- Heat transfer between adjacent segments
- Fuel consumption and flame intensity dynamics
- Visual representation of flame movement

## Usage

### Basic Simulation

Run a basic flame propagation simulation:

```bash
python flame_ring.py
```

### Customized Simulation

Run with custom duration and optional visualization:

```bash
# Run for 15 seconds with visualization
python flame_ring.py 15.0

# Run for 10 seconds without visualization
python flame_ring.py 10.0 --no-viz
```

### Interactive Examples

See `examples/` directory for additional simulation scenarios.

## How It Works

1. **Ring Topology**: The trough is divided into segments arranged in a circle, where the last segment connects back to the first (periodic boundary conditions).

2. **Flame Properties**: Each flame has characteristics like temperature, propagation speed, and fuel consumption rate.

3. **Heat Transfer**: Burning segments transfer heat to their neighbors, potentially igniting them if conditions are met.

4. **Fuel Dynamics**: Each segment has a fuel level that decreases as flames burn, affecting flame intensity and eventual extinguishing.

## Visualization Legend

- `█` `▬` `▪` `·` - Flame intensity (high to low)
- `o` - Fuel available (not burning)
- `.` - Low fuel available
- ` ` (space) - No fuel/empty

## Example Output

```
TravelingFlame: Ring-shaped Trough Simulation
==================================================
Legend: █▬▪· = flame intensity, o = fuel, . = low fuel, space = empty

Initial state:
ooo oo 
o    o 
       
o     ▬
o    o 
oo  oo 
   o   

Step 10:
ooo o▬ 
o    █ 
       
o     █
o    █ 
oo  o▬ 
   o   
```

## Technical Details

The simulation uses a discrete time-stepping approach where:
- Each segment tracks fuel level, temperature, and burning state
- Heat propagation follows simple diffusion rules
- Flame ignition requires sufficient temperature and fuel
- Ring topology ensures flames can propagate continuously around the circle

This demonstrates interesting emergent behaviors in cyclic systems and can be extended for more complex scenarios.
