# Global Freight Frenzy

## Leaderboard

### Register Your Team

https://maketheboard.com/b/sNFzFeMCDIM/self-register

### View the Leaderboard

https://maketheboard.com/b/sNFzFeMCDIM

## Example Step Function

Filename: example_step.py

## Writing a custom `step` function

```python
# my_strategy.py
from simulator import VehicleType

def step(sim_state):
    """Called by the simulator every tick."""
    tick     = sim_state.tick
    vehicles = sim_state.get_vehicles()   # read-only snapshot
    boxes    = sim_state.get_boxes()      # read-only snapshot

    if tick == 0:
        # Create a semi-truck at the Los Angeles hub
        vid = sim_state.create_vehicle(VehicleType.SemiTruck, (33.9425, -118.4081))

        # Load boxes that are at this location (costs 1 pt each)
        nearby = [bid for bid, b in boxes.items() if b["location"] == (33.9425, -118.4081)]
        sim_state.load_vehicle(vid, nearby)

        # Drive toward New York
        sim_state.move_vehicle(vid, (40.6413, -73.7781))

    for vid, v in vehicles.items():
        # When the truck arrives, unload boxes destined for this location
        if v["destination"] is None and v["cargo"]:
            loc = v["location"]
            to_drop = [bid for bid in v["cargo"] if boxes[bid]["destination"] == loc]
            if to_drop:
                sim_state.unload_vehicle(vid, to_drop)
```

---

## Vehicle types and costs

| Type        | Base cost | Per-km cost | Speed (km/h) | Capacity |
| ----------- | --------- | ----------- | ------------ | -------- |
| `SemiTruck` | 100       | 0.05        | 80           | 50       |
| `Train`     | 500       | 0.02        | 120          | 500      |
| `Airplane`  | 2000      | 0.50        | 800          | 100      |
| `CargoShip` | 1000      | 0.01        | 30           | 1000     |
| `Drone`     | 50        | 0.30        | 50           | 5        |

---

## How to Validate Your Step Function

Run with an external strategy script that exports `step(sim_state)`:

```bash
./logic-transport-sim /absolute/path/to/my_step.py --ui
```
