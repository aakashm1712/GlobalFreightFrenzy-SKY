# SimulationState API Reference

Everything available to you inside your `step(sim_state)` function.

---

## Commands

These methods **mutate** the simulation and add to `total_cost`.

---

### `create_vehicle(vehicle_type, gps_coords) → str`

Spawn a new vehicle and charge its one-time base cost.

```python
vehicle_id = sim_state.create_vehicle(VehicleType.SemiTruck, (34.05, -118.24))
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `vehicle_type` | `VehicleType` | One of the `VehicleType` enum members (see table below) |
| `gps_coords` | `(float, float)` | Starting `(latitude, longitude)` in decimal degrees |

**Returns** a unique string vehicle ID such as `"v_semitruck_1"`.

**Raises**
- `TypeError` — `vehicle_type` is not a `VehicleType` member
- `ValueError` — spawn location violates a spawn restriction (see [Spawn Rules](#spawn-rules))

---

### `move_vehicle(vehicle_id, target_gps_coords)`

Set a vehicle's destination. The physics engine advances the vehicle toward it each tick at the vehicle's speed. Movement cost accumulates per kilometre automatically.

```python
sim_state.move_vehicle(vehicle_id, (40.71, -74.01))
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `vehicle_id` | `str` | ID returned by `create_vehicle` |
| `target_gps_coords` | `(float, float)` | Target `(latitude, longitude)` |

**Raises**
- `KeyError` — `vehicle_id` does not exist

---

### `load_vehicle(vehicle_id, box_ids)`

Load a list of boxes onto a vehicle. Costs **1 point per box** loaded.

```python
sim_state.load_vehicle(vehicle_id, ["box_001", "box_002"])
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `vehicle_id` | `str` | ID returned by `create_vehicle` |
| `box_ids` | `list[str]` | Box IDs to load |

**Requirements** — per box:
- Vehicle must be within **50 metres** of the box's current location
- Box must not already be on a vehicle
- Box must not already be delivered
- Vehicle must have remaining capacity
- `Airplane` and `Drone` must be within **5 000 m** of an airport
- `CargoShip` must be within **5 000 m** of an ocean port (if `ocean_ports` are configured)

**Raises**
- `KeyError` — unknown `vehicle_id` or box ID
- `ValueError` — proximity, status, capacity, or vehicle/facility rules are violated

---

### `unload_vehicle(vehicle_id, box_ids)`

Unload boxes from a vehicle at its current location. A box is marked **delivered** automatically when unloaded within 50 metres of its destination.

```python
sim_state.unload_vehicle(vehicle_id, ["box_001"])
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `vehicle_id` | `str` | ID returned by `create_vehicle` |
| `box_ids` | `list[str]` | Box IDs to unload |

**Raises**
- `KeyError` — unknown `vehicle_id` or box ID
- `ValueError` — a box is not currently on `vehicle_id`, or vehicle/facility rules are violated

---

## Read-only Accessors

These never raise and never affect cost.

---

### `sim_state.tick → int`

Current tick number, starting at `0`.

```python
if sim_state.tick == 0:
    # one-time setup
```

---

### `sim_state.total_cost → float`

Total accumulated cost so far (vehicle creation + per-km movement + loading + terrain penalties).

---

### `sim_state.terrain_penalty → float`

The portion of `total_cost` that came from terrain violations (vehicles moving through forbidden terrain).

---

### `get_vehicles() → dict[str, dict]`

Snapshot of every vehicle. Modifying the returned dict has no effect on the simulation.

```python
vehicles = sim_state.get_vehicles()
for vid, v in vehicles.items():
    print(v["vehicle_type"], v["location"], v["cargo"])
```

Each vehicle dict contains:

| Key | Type | Description |
|-----|------|-------------|
| `id` | `str` | Vehicle ID |
| `vehicle_type` | `str` | Enum member name, e.g. `"SemiTruck"` |
| `location` | `(float, float)` | Current `(lat, lon)` |
| `destination` | `(float, float) \| None` | Current target, or `None` if idle |
| `cargo` | `list[str]` | Box IDs currently loaded |
| `total_distance_km` | `float` | Total kilometres travelled |
| `terrain_penalty_cost` | `float` | Terrain penalties charged to this vehicle |

---

### `get_boxes() → dict[str, dict]`

Snapshot of every box. Modifying the returned dict has no effect on the simulation.

```python
boxes = sim_state.get_boxes()
undelivered = [b for b in boxes.values() if not b["delivered"] and b["vehicle_id"] is None]
```

Each box dict contains:

| Key | Type | Description |
|-----|------|-------------|
| `id` | `str` | Box ID |
| `contents` | `str` | Human-readable description of the box |
| `location` | `(float, float)` | Current `(lat, lon)` |
| `destination` | `(float, float)` | Delivery target `(lat, lon)` |
| `vehicle_id` | `str \| None` | Vehicle the box is on, or `None` |
| `delivered` | `bool` | `True` once successfully delivered |

---

### `get_active_events() → list[dict]`

Returns metadata for every scenario event currently in effect at this tick.

```python
for event in sim_state.get_active_events():
    print(event["type"], "—", event["remaining_ticks"], "ticks left")
```

Each event dict contains:

| Key | Type | Always present | Description |
|-----|------|---------------|-------------|
| `id` | `int` | ✓ | Index of the event in the bootstrap config |
| `type` | `str` | ✓ | `"ground_stop_flights"`, `"traffic"`, or `"oceanic_weather"` |
| `start_tick` | `int` | ✓ | Tick the event began |
| `end_tick` | `int` | ✓ | Tick the event ends (exclusive) |
| `remaining_ticks` | `int` | ✓ | Ticks until the event expires |
| `center` | `(float, float)` | traffic / optional oceanic | Geographic centre of the affected area |
| `radius_m` | `float` | traffic / optional oceanic | Radius of the affected area in metres |
| `speed_multiplier` | `float` | traffic only | Speed fraction applied to ground vehicles (default `0.25`) |

**Event effects on movement (automatic — no action required):**

| Event type | Affected vehicles | Effect |
|------------|-------------------|--------|
| `ground_stop_flights` | `Airplane`, `Drone` | Stopped completely (speed × 0) |
| `traffic` | `SemiTruck`, `Train` | Reduced to `speed_multiplier` of normal speed within radius |
| `oceanic_weather` | `CargoShip` | Stopped completely within area (global if no radius) |

---

## Vehicle Types

Import with `from simulator import VehicleType`.

| Enum member | Base cost | Per-km cost | Speed | Capacity | Terrain |
|-------------|-----------|-------------|-------|----------|---------|
| `VehicleType.SemiTruck` | 100 | 0.05 | 80 km/h | 50 boxes | Land only |
| `VehicleType.Train` | 500 | 0.02 | 120 km/h | 500 boxes | Land only |
| `VehicleType.Airplane` | 2 000 | 0.50 | 800 km/h | 100 boxes | Unrestricted |
| `VehicleType.CargoShip` | 1 000 | 0.01 | 30 km/h | 1 000 boxes | Water only |
| `VehicleType.Drone` | 50 | 0.30 | 50 km/h | 5 boxes | Unrestricted |

Moving a vehicle through its **forbidden terrain** (e.g. a SemiTruck over water) adds an extra terrain penalty per km on top of the normal movement cost.

---

## Spawn Rules

`create_vehicle` raises `ValueError` if these are violated.

| Vehicle type | Spawn requirement |
|--------------|-------------------|
| `SemiTruck`, `Train` | Within **5 000 m** of a shipping hub |
| `Airplane`, `Drone` | Within **5 000 m** of an airport (falls back to hub locations if no airports defined in bootstrap) |
| `CargoShip` | Must be over **water** (ocean/sea), and within **5 000 m** of an ocean port when `ocean_ports` are defined |

Facility operation rules for cargo handling (`load_vehicle` / `unload_vehicle`):

- `Airplane` and `Drone` can only load/unload cargo within **5 000 m** of an airport.
- `CargoShip` can only load/unload cargo within **5 000 m** of an ocean port when `ocean_ports` are defined.

---

## Bootstrap Fields For Facility Rules

These bootstrap fields drive spawn and cargo-operation restrictions:

- `airports`: optional list of airport locations used by `Airplane` and `Drone`.
- `ocean_ports`: optional list of ocean-port locations used by `CargoShip`.

If `airports` is omitted or empty, shipping hub locations are used as fallback airport locations.
If `ocean_ports` is omitted or empty, cargo ships still must spawn over water, but no port-distance check is enforced.

Example:

```json
{
    "airports": [
        {
            "id": "airport_lax",
            "name": "Los Angeles International Airport",
            "location": { "lat": 33.9425, "lon": -118.4081 }
        }
    ],
    "ocean_ports": [
        {
            "id": "port_la",
            "name": "Port of Los Angeles",
            "location": { "lat": 33.7361, "lon": -118.2639 }
        }
    ]
}
```

To handle scenarios where not every vehicle type is valid at a given location, try each type and catch `ValueError`:

```python
import random
vehicle_types = list(VehicleType)
random.shuffle(vehicle_types)
for vtype in vehicle_types:
    try:
        vid = sim_state.create_vehicle(vtype, hub_location)
        break
    except ValueError:
        continue
```

---

## Cost Summary

| Action | Cost |
|--------|------|
| `create_vehicle(VehicleType.X, ...)` | `VehicleType.X.value.base_cost` |
| Vehicle travels 1 km | `VehicleType.X.value.per_km_cost` |
| `load_vehicle(...)` with N boxes | N × 1.0 |
| Vehicle travels 1 km in forbidden terrain | extra `VehicleType.X.value.terrain_penalty_per_km` |

Lower `total_cost` is better.

---

## Minimal Example

```python
from simulator import VehicleType

def step(sim_state):
    if sim_state.tick == 0:
        # Spawn a truck at the first box's location
        boxes = sim_state.get_boxes()
        first_box = next(iter(boxes.values()))
        vid = sim_state.create_vehicle(VehicleType.SemiTruck, first_box["location"])
        sim_state.load_vehicle(vid, [first_box["id"]])
        sim_state.move_vehicle(vid, first_box["destination"])
        return

    # Every tick: unload delivered cargo, then keep moving
    vehicles = sim_state.get_vehicles()
    boxes = sim_state.get_boxes()
    for vid, vehicle in vehicles.items():
        if vehicle["destination"] is not None:
            continue  # still en route
        deliverable = [
            bid for bid in vehicle["cargo"]
            if boxes[bid]["destination"] == vehicle["location"]
        ]
        if deliverable:
            sim_state.unload_vehicle(vid, deliverable)

    # Optionally check for active events
    for event in sim_state.get_active_events():
        if event["type"] == "ground_stop_flights":
            print(f"Ground stop in effect — {event['remaining_ticks']} ticks left")
```
...
