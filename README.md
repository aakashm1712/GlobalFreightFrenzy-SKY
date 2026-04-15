# Global Freight Frenzy

## Communication

We will be using a Discord server for communication for the event. If you need any help from organizers or want to help one another, please join the link below. 

https://discord.gg/8Y9dqk59

While we fully support students helping one another with issues, please do not share any solutions or files. 

## Forking the Repository

In order to maintain secrecy among contestants, we will be using GitHub for submissions and enforcing contestants to 'fork' the repository and then 'submit' their work by creating a PR against the main branch.

1. Fork this repository on GitHub

- Go to https://github.com/Meta-Logic-Corp/GlobalFreightFrenzy
- Click **Fork** in the top-right corner
- Create the fork under your own GitHub account

2. Install Git LFS

```bash
git lfs install
```

3. Clone your fork

```bash
git clone https://github.com/<your-username>/GlobalFreightFrenzy.git
cd GlobalFreightFrenzy
```

4. Add the original repo as `upstream`

```bash
git remote add upstream https://github.com/Meta-Logic-Corp/GlobalFreightFrenzy.git
git remote -v
```

5. Create a feature branch

```bash
git checkout -b feature/my-change
```

6. Commit and push your changes to your fork

```bash
git add .
git commit -m "Describe your change"
git push -u origin feature/my-change
```

## Creating a Pull Request

1. Open your fork on GitHub and switch to your branch.
2. Click **Compare & pull request**.
3. Set:

- **base repository:** `Meta-Logic-Corp/GlobalFreightFrenzy`
- **base branch:** `main` (or the target branch requested by maintainers)
- **head repository:** your fork
- **compare branch:** `feature/my-change`

4. Add a clear title and description of what changed and why.
5. Click **Create pull request**.
6. Respond to review feedback by pushing additional commits to the same branch.

## Reporting Issues

If you find a bug, simulation inconsistency, or unclear behavior, please open an issue in the main repository:

https://github.com/Meta-Logic-Corp/GlobalFreightFrenzy/issues

When creating an issue, include:

1. A clear title that describes the problem.
2. Steps to reproduce the issue.
3. Expected behavior vs. actual behavior.
4. Relevant logs, error messages, or screenshots.
5. Environment details (OS, Python version, and command used).

Before opening a new issue, search existing issues to avoid duplicates.

## Leaderboard

### Register Your Team

https://maketheboard.com/b/sNFzFeMCDIM/self-register

### View the Leaderboard

https://maketheboard.com/b/sNFzFeMCDIM

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

        # Optional: inspect currently active scenario events
        active_events = sim_state.get_active_events()
        for event in active_events:
          print(event["type"], event["remaining_ticks"])
```

`create_vehicle(...)` raises `ValueError` when spawn restrictions are violated:

- `SemiTruck` and `Train` must be within 5 km of a shipping hub.
- `Airplane` and `Drone` must be within 5 km of an airport.
- `CargoShip` must be spawned over water (ocean/sea) and within 5 km of an ocean port (when `ocean_ports` are configured).

If no explicit `airports` list is provided in bootstrap JSON, hub locations are
used as airport spawn points.

Cargo handling facility rules:

- `Airplane` and `Drone` can only load/unload cargo within 5 km of an airport.
- `CargoShip` can only load/unload cargo within 5 km of an ocean port (when `ocean_ports` are configured).

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

## Scenario presets

Three ready-to-run scenario files are available in `scenarios/`:

- `scenario_1_us_simple.json`: 50 boxes, no events, 2 US origins to 2 US destinations.
- `scenario_2_global_heavy.json`: 500 boxes, one event of each type.
- `scenario_3_event_storm.json`: 1000 boxes, 10 events, many world destinations.

Run any scenario with the built-in strategy:

```bash
./logic-transport-sim /absolute/path/to/my_step.py --ui --bootstrap scenarios/scenario_1_us_simple.json
./logic-transport-sim /absolute/path/to/my_step.py --ui --bootstrap scenarios/scenario_2_global_heavy.json
./logic-transport-sim  /absolute/path/to/my_step.py --ui --bootstrap scenarios/scenario_3_event_storm.json
```
