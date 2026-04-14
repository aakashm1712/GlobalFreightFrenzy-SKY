"""Example step function for the LogicTransportationSimulator.

Strategy: Greedy one-vehicle-per-hub delivery using random vehicle types.

* **Tick 0** – one randomly chosen vehicle type is created at each shipping hub and immediately
  loaded with all boxes located there.
* **Every tick** – each stationary truck:
    1. Unloads any cargo that has reached its destination (within 50 m).
    2. Loads any unloaded boxes at its current location.
    3. Drives toward the first remaining box's destination.

Trucks with multiple boxes destined for different locations complete one
delivery at a time, then reposition for the next.
"""

from simulator import VehicleType, haversine_distance_meters
import random

_PROXIMITY_M = 50.0


def step(sim_state):
    """Called by the simulator each tick.

    Args:
        sim_state: A :class:`~simulator.SimulationState` instance.  State may
                   only be modified through its public command methods.
    """
    tick = sim_state.tick

    # ── Tick 0: spawn one random vehicle at every unique hub location ────
    if tick == 0:
        boxes = sim_state.get_boxes()
        seen_hubs = set()
        vehicle_types = list(VehicleType)
        for box in boxes.values():
            loc = box["location"]
            if loc not in seen_hubs:
                seen_hubs.add(loc)
                sim_state.create_vehicle(random.choice(vehicle_types), loc)

    # ── Every tick: manage each vehicle ──────────────────────────────────
    vehicles = sim_state.get_vehicles()
    boxes = sim_state.get_boxes()

    for vid, vehicle in vehicles.items():
        loc = vehicle["location"]
        config = VehicleType[vehicle["vehicle_type"]].value
        remaining_capacity = config.capacity - len(vehicle["cargo"])
        has_capacity = remaining_capacity > 0

        # Skip vehicles that are still en route.
        if vehicle["destination"] is not None:
            continue

        # 1. Unload boxes whose destination is at (or within 50 m of) here.
        deliverable = [
            bid
            for bid in vehicle["cargo"]
            if haversine_distance_meters(loc, boxes[bid]["destination"]) <= _PROXIMITY_M
        ]
        if deliverable:
            sim_state.unload_vehicle(vid, deliverable)
            boxes = sim_state.get_boxes()  # refresh after unload

        # 2. Load any unloaded, undelivered boxes at this location.
        loadable = [
            bid
            for bid, box in boxes.items()
            if not box["delivered"]
            and box["vehicle_id"] is None
            and haversine_distance_meters(loc, box["location"]) <= _PROXIMITY_M
        ]
        if loadable and has_capacity:
            sim_state.load_vehicle(vid, loadable[:remaining_capacity])
            boxes = sim_state.get_boxes()  # refresh after load

        # 3. Head toward the first cargo box's destination (if any remain).
        vehicles = sim_state.get_vehicles()  # refresh cargo list
        vehicle = vehicles[vid]
        if vehicle["cargo"]:
            next_dest = boxes[vehicle["cargo"][0]]["destination"]
            sim_state.move_vehicle(vid, next_dest)
