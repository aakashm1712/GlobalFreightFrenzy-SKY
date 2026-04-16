from collections import defaultdict
from heapq import heappush, heappop
from math import ceil, radians, degrees, sin, cos, asin, atan2
from simulator import VehicleType, haversine_distance_meters  # type: ignore
from simulator.physics import is_over_land  # type: ignore

_world = {}

# ------------------------------------------------------------
# Cost rules
# ------------------------------------------------------------
COSTS = {
    "semi": {"base": 100.0, "per_km": 0.05},
    "train": {"base": 500.0, "per_km": 0.02},
    "plane": {"base": 2000.0, "per_km": 0.50},
    "ship": {"base": 1000.0, "per_km": 0.01},
    "drone": {"base": 50.0, "per_km": 0.30},
}

LAND_STEP_KM = 80.0
ARRIVAL_THRESHOLD_M = 25000.0
MAX_ROUTE_STEPS = 1200
HUB_MATCH_THRESHOLD_M = 10000.0

NA_HUBS = {
    "hub_seattle",
    "hub_la",
    "hub_dallas",
    "hub_mexico_city",
    "hub_miami",
    "hub_nyc",
    "hub_chicago",
    "hub_toronto",
}

OLD_WORLD_HUBS = {
    "hub_london",
    "hub_frankfurt",
    "hub_nairobi",
    "hub_johannesburg",
    "hub_dubai",
    "hub_mumbai",
    "hub_singapore",
    "hub_tokyo",
}

HUB_TO_REGION = {
    "hub_seattle": "north_america",
    "hub_la": "north_america",
    "hub_dallas": "north_america",
    "hub_mexico_city": "north_america",
    "hub_miami": "north_america",
    "hub_nyc": "north_america",
    "hub_chicago": "north_america",
    "hub_toronto": "north_america",

    "hub_london": "old_world",
    "hub_frankfurt": "old_world",
    "hub_nairobi": "old_world",
    "hub_johannesburg": "old_world",
    "hub_dubai": "old_world",
    "hub_mumbai": "old_world",
    "hub_singapore": "old_world",
    "hub_tokyo": "old_world",

    "hub_sao_paulo": "south_america",
    "hub_sydney": "australia",
}

SA_SINGLE = {"hub_sao_paulo"}
AU_SINGLE = {"hub_sydney"}


def _coord(detail):
    """Convert a detail object with location dict into (lat, lon)."""
    return (detail["location"]["lat"], detail["location"]["lon"])


def _km_between(a, b):
    return haversine_distance_meters(a, b) / 1000.0


def _cost(mode, distance_km):
    return round(COSTS[mode]["base"] + COSTS[mode]["per_km"] * distance_km, 2)


def _group_of_hub(hub_id):
    return HUB_TO_REGION.get(hub_id)

def _is_intra_land_pair(origin_hub_id, destination_hub_id):
    """
    Only allow the intra-land phase for:
    - North America internal transfers
    - Europe+Asia+Africa internal transfers
    Skip São Paulo and Sydney mixed routing for now.
    """
    g1 = _group_of_hub(origin_hub_id)
    g2 = _group_of_hub(destination_hub_id)

    if g1 is None or g2 is None:
        return False

    if g1 != g2:
        return False

    return g1 in {"north_america", "old_world"}


def _nearest_facility(origin_coord, facilities):
    """
    Return the closest facility to origin_coord.
    """
    if not facilities:
        return None

    best = None
    best_dist = float("inf")

    for facility in facilities.values():
        dist_m = haversine_distance_meters(origin_coord, facility["coord"])
        if dist_m < best_dist:
            best_dist = dist_m
            best = facility

    return {
        "id": best["id"],
        "name": best["name"],
        "coord": best["coord"],
        "distance_m": round(best_dist, 2),
    }


def _find_closest_hub(coord, hubs, max_match_m=1000):
    """
    Match a box location/destination to the nearest hub.
    This is safer than exact float equality.
    """
    best_hub = None
    best_dist = float("inf")

    for hub in hubs.values():
        dist_m = haversine_distance_meters(coord, hub["coord"])
        if dist_m < best_dist:
            best_dist = dist_m
            best_hub = hub

    if best_hub is None or best_dist > max_match_m:
        return None

    return {
        "id": best_hub["id"],
        "name": best_hub["name"],
        "coord": best_hub["coord"],
        "distance_m": round(best_dist, 2),
    }


def _bearing_deg(start, end):
    lat1 = radians(start[0])
    lon1 = radians(start[1])
    lat2 = radians(end[0])
    lon2 = radians(end[1])

    dlon = lon2 - lon1

    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)

    bearing = degrees(atan2(x, y))
    return (bearing + 360.0) % 360.0


def _destination_point(start, bearing_deg, distance_km):
    """
    Move from a start point by bearing and distance.
    """
    earth_radius_km = 6371.0

    lat1 = radians(start[0])
    lon1 = radians(start[1])
    brng = radians(bearing_deg)
    angular_distance = distance_km / earth_radius_km

    lat2 = asin(
        sin(lat1) * cos(angular_distance)
        + cos(lat1) * sin(angular_distance) * cos(brng)
    )

    lon2 = lon1 + atan2(
        sin(brng) * sin(angular_distance) * cos(lat1),
        cos(angular_distance) - sin(lat1) * sin(lat2),
    )

    return (degrees(lat2), degrees(lon2))


def _move_toward_km(current, target, step_km):
    """
    Simple great-circle step toward target.
    """
    dist_km = _km_between(current, target)
    if dist_km <= step_km:
        return target

    bearing = _bearing_deg(current, target)
    return _destination_point(current, bearing, step_km)


def _choose_land_step(current, target, step_km=LAND_STEP_KM):
    """
    Project next point.
    If it lands on water, change angle and choose the best land point.
    """
    direct_next = _move_toward_km(current, target, step_km)
    if is_over_land(direct_next):
        return direct_next

    base_bearing = _bearing_deg(current, target)
    current_remain = haversine_distance_meters(current, target)

    step_sizes = [step_km, step_km * 0.5, step_km * 0.25]
    angle_offsets = [
        5, -5, 10, -10, 15, -15, 20, -20, 30, -30, 45, -45,
        60, -60, 75, -75, 90, -90, 110, -110, 135, -135, 160, -160, 180
    ]

    best = None
    best_score = float("inf")

    for cur_step in step_sizes:
        for offset in angle_offsets:
            candidate = _destination_point(current, base_bearing + offset, cur_step)

            if not is_over_land(candidate):
                continue

            remain = haversine_distance_meters(candidate, target)

            # prefer progress, but allow detours
            if remain > current_remain + 20000:
                continue

            score = remain + abs(offset) * 500.0
            if score < best_score:
                best_score = score
                best = candidate

        if best is not None:
            return best

    return None


def _build_land_path(start, end, step_km=LAND_STEP_KM):
    """
    Build a land-only heuristic path by repeatedly projecting
    the next point and correcting if it goes over water.
    """
    if not is_over_land(start):
        return None
    if not is_over_land(end):
        return None

    path_points = [start]
    current = start
    total_km = 0.0

    for _ in range(MAX_ROUTE_STEPS):
        remaining_m = haversine_distance_meters(current, end)
        if remaining_m <= ARRIVAL_THRESHOLD_M:
            total_km += remaining_m / 1000.0
            path_points.append(end)
            return {
                "distance_km": round(total_km, 2),
                "points": path_points,
            }

        next_point = _choose_land_step(current, end, step_km)
        if next_point is None:
            return None

        segment_km = _km_between(current, next_point)
        if segment_km < 0.001:
            return None

        total_km += segment_km
        current = next_point
        path_points.append(current)

    return None


def _ground_transfer_path(start, end):
    path = _build_land_path(start, end)
    if path is not None:
        return path

    distance_km = _km_between(start, end)
    return {
        "distance_km": round(distance_km, 2),
        "points": [start, end],
        "fallback": "direct_ground_transfer",
    }


def _build_air_combo_plan(origin_hub, destination_hub, airports):
    """
    Air = semi to closest airport + plane + semi to destination hub
    """
    origin_airport = _nearest_facility(origin_hub["coord"], airports)
    destination_airport = _nearest_facility(destination_hub["coord"], airports)

    if origin_airport is None or destination_airport is None:
        return None

    ground_leg_1 = _build_land_path(origin_hub["coord"], origin_airport["coord"])
    if ground_leg_1 is None:
        return None

    air_leg_km = _km_between(origin_airport["coord"], destination_airport["coord"])

    ground_leg_2 = _build_land_path(destination_airport["coord"], destination_hub["coord"])
    if ground_leg_2 is None:
        return None

    semi_1_cost = _cost("semi", ground_leg_1["distance_km"])
    plane_cost = _cost("plane", air_leg_km)
    semi_2_cost = _cost("semi", ground_leg_2["distance_km"])

    return {
        "origin_airport": origin_airport,
        "destination_airport": destination_airport,
        "ground_leg_1_km": round(ground_leg_1["distance_km"], 2),
        "air_leg_km": round(air_leg_km, 2),
        "ground_leg_2_km": round(ground_leg_2["distance_km"], 2),
        "total_km": round(
            ground_leg_1["distance_km"] + air_leg_km + ground_leg_2["distance_km"], 2
        ),
        "cost": round(semi_1_cost + plane_cost + semi_2_cost, 2),
        "ground_path_to_airport": ground_leg_1["points"],
        "ground_path_from_airport": ground_leg_2["points"],
    }


def _build_pair_plan(origin_hub, destination_hub, airports, ocean_ports):
    origin_region = _group_of_hub(origin_hub["id"])
    destination_region = _group_of_hub(destination_hub["id"])

    # Intra-region: keep your richer logic
    if origin_region == destination_region:
        land_path = _build_land_path(origin_hub["coord"], destination_hub["coord"])
        air_combo = _build_air_combo_plan(origin_hub, destination_hub, airports)

        plan = {
            "origin_hub_id": origin_hub["id"],
            "destination_hub_id": destination_hub["id"],
            "origin_hub_name": origin_hub["name"],
            "destination_hub_name": destination_hub["name"],
            "group": origin_region,
            "land_path": None,
            "semi": None,
            "train": None,
            "air_combo": air_combo,
            "water_combo": None,
            "best_mode": None,
            "best_cost": None,
        }

        if land_path is not None:
            plan["land_path"] = land_path
            plan["semi"] = {
                "distance_km": land_path["distance_km"],
                "cost": _cost("semi", land_path["distance_km"]),
            }
            plan["train"] = {
                "distance_km": land_path["distance_km"],
                "cost": _cost("train", land_path["distance_km"]),
            }

        candidates = []
        if plan["semi"] is not None:
            candidates.append((plan["semi"]["cost"], "semi"))
        if plan["train"] is not None:
            candidates.append((plan["train"]["cost"], "train"))
        if plan["air_combo"] is not None:
            candidates.append((plan["air_combo"]["cost"], "air_combo"))

        if candidates:
            best_cost, best_mode = min(candidates)
            plan["best_mode"] = best_mode
            plan["best_cost"] = best_cost

        return plan

    # Inter-region disabled — return None to skip cross-continental routing.
    return None

    # Inter-region: skip ocean ports and use the airport transfer chain.
    air_combo = _build_air_combo_plan(origin_hub, destination_hub, airports)
    if air_combo is None:
        return None

    return {
        "origin_hub_id": origin_hub["id"],
        "destination_hub_id": destination_hub["id"],
        "origin_hub_name": origin_hub["name"],
        "destination_hub_name": destination_hub["name"],
        "group": origin_region,
        "land_path": None,
        "semi": None,
        "train": None,
        "air_combo": air_combo,
        "water_combo": None,
        "best_mode": "air_combo",
        "best_cost": air_combo["cost"],
        "kind": "inter_region_air",
    }
def _build_world(sim_state):
    """
    Build a static snapshot of:
    - all hubs
    - all airports
    - all boxes with current location + destination
    - per-hub box groupings
    - nearest airport for each hub
    - distances from each hub to every other hub
    - intra-land pair plans
    """
    global _world

    hub_details = sim_state.get_shipping_hub_details()
    airport_details = sim_state.get_airport_details()
    boxes = sim_state.get_boxes()

    # -----------------------------
    # 1) Facilities
    # -----------------------------
    hubs = {}
    for hub in hub_details:
        coord = _coord(hub)
        hubs[hub["id"]] = {
            "id": hub["id"],
            "name": hub["name"],
            "coord": coord,
            "group": _group_of_hub(hub["id"]),
            "box_ids_here": [],
            "boxes_here": [],
            "destination_groups": {},   # dest_hub_id -> list of box summaries
            "nearest_airport": None,
            "nearest_ocean_port": None,
            "dist_to_other_hubs_m": {}, # other_hub_id -> meters
        }

    airports = {}
    for airport in airport_details:
        coord = _coord(airport)
        airports[airport["id"]] = {
            "id": airport["id"],
            "name": airport["name"],
            "coord": coord,
        }

    ocean_ports = {}

    # -----------------------------
    # 2) Hub -> nearest airport. Ocean ports are intentionally ignored.
    # -----------------------------
    for hub in hubs.values():
        hub["nearest_airport"] = _nearest_facility(hub["coord"], airports)
        hub["nearest_ocean_port"] = None

    # -----------------------------
    # 3) Hub -> other hub distances
    # -----------------------------
    for from_hub in hubs.values():
        for to_hub in hubs.values():
            if from_hub["id"] == to_hub["id"]:
                continue

            dist_m = haversine_distance_meters(from_hub["coord"], to_hub["coord"])
            from_hub["dist_to_other_hubs_m"][to_hub["id"]] = round(dist_m, 2)

    # -----------------------------
    # 3.5) Hub pair planning
    # -----------------------------
    pair_plans = {}

    hub_ids = list(hubs.keys())
    for origin_hub_id in hub_ids:
        pair_plans[origin_hub_id] = {}

        for destination_hub_id in hub_ids:
            if origin_hub_id == destination_hub_id:
                continue

            plan = _build_pair_plan(
                hubs[origin_hub_id],
                hubs[destination_hub_id],
                airports,
                ocean_ports,
            )

            if plan is not None:
                pair_plans[origin_hub_id][destination_hub_id] = plan

    # -----------------------------
    # 4) Boxes + hub-wise grouping
    # -----------------------------
    box_records = {}

    for box_id, box in boxes.items():
        current_location = box["location"]
        destination = box["destination"]

        origin_hub = _find_closest_hub(current_location, hubs)
        destination_hub = _find_closest_hub(destination, hubs)

        pair_plan = None
        plan_status = "unplanned"

        if origin_hub and destination_hub:
            if origin_hub["id"] == destination_hub["id"]:
                plan_status = "local_same_hub"
            else:
                pair_plan = pair_plans.get(origin_hub["id"], {}).get(destination_hub["id"])
                if pair_plan and pair_plan["best_mode"] is not None:
                    plan_status = "planned"
                else:
                    plan_status = "no_valid_route_found"
        else:
            plan_status = "unknown_hub_match"

        box_info = {
            "id": box["id"],
            "contents": box["contents"],
            "current_location": current_location,
            "destination": destination,
            "origin_hub_id": origin_hub["id"] if origin_hub else None,
            "origin_hub_name": origin_hub["name"] if origin_hub else None,
            "destination_hub_id": destination_hub["id"] if destination_hub else None,
            "destination_hub_name": destination_hub["name"] if destination_hub else None,
            "vehicle_id": box["vehicle_id"],
            "delivered": box["delivered"],
            "plan_status": plan_status,
            "pair_plan": {
                "best_mode": pair_plan["best_mode"],
                "best_cost": pair_plan["best_cost"],
            } if pair_plan else None,
        }

        box_records[box_id] = box_info

        # Group this box under its current/origin hub
        if origin_hub:
            hub_entry = hubs[origin_hub["id"]]
            hub_entry["box_ids_here"].append(box_id)
            hub_entry["boxes_here"].append(box_info)

            dest_key = (
                destination_hub["id"] if destination_hub else "unknown_destination"
            )
            if dest_key not in hub_entry["destination_groups"]:
                hub_entry["destination_groups"][dest_key] = []

            hub_entry["destination_groups"][dest_key].append(box_info)

    # -----------------------------
    # 5) Final static world object
    # -----------------------------
    _world = {
        "hubs": hubs,
        "airports": airports,
        "ocean_ports": ocean_ports,
        "boxes": box_records,
        "pair_plans": pair_plans,
    }

    return _world


def _format_point(pt):
    return f"({pt[0]:.4f}, {pt[1]:.4f})"


def _preview_points(points, max_points=6):
    """
    Show a readable preview of route points without dumping hundreds of coordinates.
    """
    if not points:
        return "none"

    if len(points) <= max_points:
        return " -> ".join(_format_point(p) for p in points)

    head_count = max_points // 2
    tail_count = max_points - head_count

    head = points[:head_count]
    tail = points[-tail_count:]

    return (
        " -> ".join(_format_point(p) for p in head)
        + f" -> ... ({len(points) - len(head) - len(tail)} more) ... -> "
        + " -> ".join(_format_point(p) for p in tail)
    )


def _print_world_summary():
    """Debug print with hub summary + route comparisons."""
    print("\nSTATIC WORLD MODEL BUILT")
    print("========================")
    print(f"Hubs: {len(_world['hubs'])}")
    print(f"Airports: {len(_world['airports'])}")
    print(f"Ocean ports: {len(_world['ocean_ports'])}")
    print(f"Boxes: {len(_world['boxes'])}")
    print("")

    pair_plans = _world.get("pair_plans", {})

    for hub in _world["hubs"].values():
        print(f"{hub['name']} ({hub['id']})")
        print(f"  coord: {hub['coord']}")
        print(f"  boxes here: {len(hub['box_ids_here'])}")

        if hub["nearest_airport"]:
            print(
                f"  nearest airport: {hub['nearest_airport']['name']} "
                f"({hub['nearest_airport']['distance_m']} m)"
            )
        else:
            print("  nearest airport: none")

        if hub["nearest_ocean_port"]:
            print(
                f"  nearest ocean port: {hub['nearest_ocean_port']['name']} "
                f"({hub['nearest_ocean_port']['distance_m']} m)"
            )
        else:
            print("  nearest ocean port: none")

        print("  destination groups:")
        for dest_hub_id, box_list in hub["destination_groups"].items():
            print(f"    -> {dest_hub_id}: {len(box_list)} boxes")

            # Skip non-hub / unknown destinations
            if dest_hub_id == "unknown_destination":
                print("       considerations: unknown destination hub match")
                continue

            plan = pair_plans.get(hub["id"], {}).get(dest_hub_id)

            if plan and plan.get("kind") == "international_fast":
                print(f"       considerations: international aggregated plan selected")
                print(f"         best : {plan.get('best_mode')} @ ${plan.get('best_cost')}")
                continue

            if not plan:
                print("       considerations: no intra-land plan stored for this pair")
                continue

            print("       considerations:")
            print(f"         group: {plan.get('group')}")

            if plan.get("semi"):
                print(
                    f"         semi : cost=${plan['semi']['cost']} | "
                    f"distance={plan['semi']['distance_km']} km"
                )
            else:
                print("         semi : unavailable")

            if plan.get("train"):
                print(
                    f"         train: cost=${plan['train']['cost']} | "
                    f"distance={plan['train']['distance_km']} km"
                )
            else:
                print("         train: unavailable")

            if plan.get("air_combo"):
                air = plan["air_combo"]
                print(
                    f"         air  : cost=${air['cost']} | "
                    f"total={air['total_km']} km | "
                    f"ground1={air['ground_leg_1_km']} km + "
                    f"air={air['air_leg_km']} km + "
                    f"ground2={air['ground_leg_2_km']} km"
                )
            else:
                print("         air  : unavailable")

            print(
                f"         best : {plan.get('best_mode')} "
                f"@ ${plan.get('best_cost')}"
            )

            if plan.get("water_combo"):
                water = plan["water_combo"]
                print(
                    f"         water: cost=${water['cost']} | "
                    f"total={water['total_km']} km | "
                    f"ground1={water['origin_ground_km']} km + "
                    f"ship={water['ship_km']} km + "
                    f"ground2={water['destination_ground_km']} km"
                )
            else:
                print("         water: unavailable")

            print("       routes:")

            if plan.get("land_path"):
                land_points = plan["land_path"]["points"]
                print(
                    f"         land route ({len(land_points)} pts, "
                    f"{plan['land_path']['distance_km']} km):"
                )
                print(f"           {_preview_points(land_points)}")
            else:
                print("         land route: unavailable")

            if plan.get("air_combo"):
                air = plan["air_combo"]
                print("         air combo route:")
                print(
                    f"           hub -> airport: {air['origin_airport']['name']} "
                    f"{_format_point(air['origin_airport']['coord'])}"
                )
                print(
                    f"           airport -> airport: "
                    f"{air['origin_airport']['name']} -> {air['destination_airport']['name']}"
                )
                print(
                    f"           airport -> hub: {air['destination_airport']['name']} "
                    f"-> {plan['destination_hub_name']}"
                )

                print("           ground path to airport:")
                print(f"             {_preview_points(air['ground_path_to_airport'])}")

                print("           ground path from airport:")
                print(f"             {_preview_points(air['ground_path_from_airport'])}")
            else:
                print("         air combo route: unavailable")
            
            if plan.get("water_combo"):
                water = plan["water_combo"]
                print("         water combo route:")
                print(
                    f"           hub -> port: {water['origin_port']['name']} "
                    f"{_format_point(water['origin_port']['coord'])}"
                )
                print(
                    f"           port -> port: "
                    f"{water['origin_port']['name']} -> {water['destination_port']['name']}"
                )
                print(
                    f"           port -> hub: {water['destination_port']['name']} "
                    f"-> {plan['destination_hub_name']}"
                )

                print("           ground path to port:")
                print(f"             {_preview_points(water['origin_ground_route_points'])}")

                print("           ship path:")
                print(f"             {_preview_points(water['ship_route_points'])}")

                print("           ground path from port:")
                print(f"             {_preview_points(water['destination_ground_route_points'])}")
            else:
                print("         water combo route: unavailable")

        print("")


PROXIMITY_M = 50.0


def _chunked(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _mode_to_vehicle_type(mode):
    if mode == "semi":
        return VehicleType.SemiTruck
    if mode == "train":
        return VehicleType.Train
    if mode == "air_combo":
        return VehicleType.Airplane
    if mode == "ship":
        return VehicleType.CargoShip
    return None


def _vehicle_capacity(vehicle_type):
    return vehicle_type.value.capacity


def _load_specific_boxes(sim_state, vehicle_id, wanted_box_ids, boxes, max_count=None):
    loadable = [
        bid for bid in wanted_box_ids
        if bid in boxes
        and not boxes[bid]["delivered"]
        and boxes[bid]["vehicle_id"] is None
    ]

    if max_count is not None:
        loadable = loadable[:max_count]

    if not loadable:
        return []

    try:
        sim_state.load_vehicle(vehicle_id, loadable)
        return loadable
    except ValueError:
        return []


def _unload_specific_boxes(sim_state, vehicle_id, box_ids):
    if not box_ids:
        return False
    try:
        sim_state.unload_vehicle(vehicle_id, box_ids)
        return True
    except ValueError:
        return False


def _unload_deliverable_boxes(sim_state, vehicle_id, vehicle, boxes):
    deliverable = [
        bid for bid in vehicle["cargo"]
        if bid in boxes
        and haversine_distance_meters(vehicle["location"], boxes[bid]["destination"]) <= PROXIMITY_M
    ]
    _unload_specific_boxes(sim_state, vehicle_id, deliverable)
    return deliverable


def _send_next_waypoint(sim_state, vehicle_id, vehicle_state, current_location):
    route_points = vehicle_state.get("route_points") or []
    idx = vehicle_state.get("next_idx", 0)

    while idx < len(route_points):
        if haversine_distance_meters(current_location, route_points[idx]) <= PROXIMITY_M:
            idx += 1
        else:
            break

    vehicle_state["next_idx"] = idx

    if idx >= len(route_points):
        return False

    sim_state.move_vehicle(vehicle_id, route_points[idx])
    vehicle_state["next_idx"] = idx + 1
    return True


def _start_direct_job(sim_state, runtime, origin_hub, destination_hub_id, plan, box_batch):
    mode = plan["best_mode"]
    vehicle_type = _mode_to_vehicle_type(mode)

    if vehicle_type is None:
        return

    try:
        vehicle_id = sim_state.create_vehicle(vehicle_type, origin_hub["coord"])
    except ValueError:
        return

    boxes = sim_state.get_boxes()
    _load_specific_boxes(sim_state, vehicle_id, box_batch, boxes, _vehicle_capacity(vehicle_type))
    vehicles = sim_state.get_vehicles()

    if vehicle_id not in vehicles:
        return

    job_id = runtime["next_job_id"]
    runtime["next_job_id"] += 1

    runtime["jobs"][job_id] = {
        "id": job_id,
        "kind": "direct",
        "mode": mode,
        "origin_hub_id": origin_hub["id"],
        "destination_hub_id": destination_hub_id,
        "box_ids": list(box_batch),
        "done": False,
    }

    runtime["vehicles"][vehicle_id] = {
        "job_id": job_id,
        "phase": "direct",
        "route_points": plan["land_path"]["points"] if plan.get("land_path") else [],
        "next_idx": 0,
        "done": False,
    }

    _send_next_waypoint(
        sim_state,
        vehicle_id,
        runtime["vehicles"][vehicle_id],
        vehicles[vehicle_id]["location"],
    )


def _start_air_job(sim_state, runtime, origin_hub, destination_hub_id, plan, box_batch):
    air = plan.get("air_combo")
    if not air:
        return

    job_id = runtime["next_job_id"]
    runtime["next_job_id"] += 1

    # origin truck: hub -> origin airport, loaded
    try:
        origin_ground_vid = sim_state.create_vehicle(VehicleType.SemiTruck, origin_hub["coord"])
    except ValueError:
        return

    boxes = sim_state.get_boxes()
    loaded = _load_specific_boxes(sim_state, origin_ground_vid, box_batch, boxes, VehicleType.SemiTruck.value.capacity)
    if not loaded:
        return

    # destination truck: destination hub -> destination airport, empty reposition
    dest_hub = _world["hubs"][destination_hub_id]
    try:
        dest_ground_vid = sim_state.create_vehicle(VehicleType.SemiTruck, dest_hub["coord"])
    except ValueError:
        return

    runtime["jobs"][job_id] = {
        "id": job_id,
        "kind": "air_combo",
        "mode": "air_combo",
        "origin_hub_id": origin_hub["id"],
        "destination_hub_id": destination_hub_id,
        "box_ids": list(box_batch),
        "origin_airport_coord": air["origin_airport"]["coord"],
        "destination_airport_coord": air["destination_airport"]["coord"],
        "forward_dest_route": air["ground_path_from_airport"],
        "plane_vid": None,
        "origin_ground_vid": origin_ground_vid,
        "dest_ground_vid": dest_ground_vid,
        "done": False,
    }

    runtime["vehicles"][origin_ground_vid] = {
        "job_id": job_id,
        "phase": "air_origin_ground",
        "route_points": air["ground_path_to_airport"],
        "next_idx": 0,
        "done": False,
    }

    runtime["vehicles"][dest_ground_vid] = {
        "job_id": job_id,
        "phase": "air_dest_ground",
        "route_points": list(reversed(air["ground_path_from_airport"])),
        "next_idx": 0,
        "waiting_at_airport": False,
        "carrying_final_leg": False,
        "done": False,
    }

    vehicles = sim_state.get_vehicles()

    if origin_ground_vid in vehicles:
        _send_next_waypoint(
            sim_state,
            origin_ground_vid,
            runtime["vehicles"][origin_ground_vid],
            vehicles[origin_ground_vid]["location"],
        )

    if dest_ground_vid in vehicles:
        _send_next_waypoint(
            sim_state,
            dest_ground_vid,
            runtime["vehicles"][dest_ground_vid],
            vehicles[dest_ground_vid]["location"],
        )


def _init_runtime(sim_state):
    runtime = {
        "next_job_id": 1,
        "jobs": {},
        "vehicles": {},
    }

    for origin_hub in _world["hubs"].values():
        origin_id = origin_hub["id"]

        for destination_hub_id, box_list in origin_hub["destination_groups"].items():
            if destination_hub_id == "unknown_destination":
                continue

            plan = _world.get("pair_plans", {}).get(origin_id, {}).get(destination_hub_id)
            if not plan or not plan.get("best_mode"):
                continue

            planned_box_ids = [
                b["id"]
                for b in box_list
                if _world["boxes"][b["id"]]["plan_status"] == "planned"
            ]

            if not planned_box_ids:
                continue

            if plan["best_mode"] == "semi":
                batch_size = VehicleType.SemiTruck.value.capacity
                for batch in _chunked(planned_box_ids, batch_size):
                    _start_direct_job(sim_state, runtime, origin_hub, destination_hub_id, plan, batch)

            elif plan["best_mode"] == "train":
                batch_size = VehicleType.Train.value.capacity
                for batch in _chunked(planned_box_ids, batch_size):
                    _start_direct_job(sim_state, runtime, origin_hub, destination_hub_id, plan, batch)

            elif plan["best_mode"] == "air_combo":
                # use semi capacity because the air chain uses semis on the first and last leg
                batch_size = VehicleType.SemiTruck.value.capacity
                for batch in _chunked(planned_box_ids, batch_size):
                    _start_air_job(sim_state, runtime, origin_hub, destination_hub_id, plan, batch)

    _world["runtime"] = runtime


def _try_spawn_and_launch_plane(sim_state, runtime, job):
    if job.get("plane_vid"):
        return

    try:
        plane_vid = sim_state.create_vehicle(VehicleType.Airplane, job["origin_airport_coord"])
    except ValueError:
        return

    boxes = sim_state.get_boxes()
    loaded = _load_specific_boxes(
        sim_state,
        plane_vid,
        job["box_ids"],
        boxes,
        VehicleType.Airplane.value.capacity,
    )

    if not loaded:
        return

    runtime["vehicles"][plane_vid] = {
        "job_id": job["id"],
        "phase": "air_plane",
        "route_points": [],
        "next_idx": 0,
        "done": False,
    }

    job["plane_vid"] = plane_vid
    sim_state.move_vehicle(plane_vid, job["destination_airport_coord"])


def _try_start_final_ground_leg(sim_state, runtime, job):
    dest_vid = job["dest_ground_vid"]
    vehicles = sim_state.get_vehicles()
    boxes = sim_state.get_boxes()

    if dest_vid not in vehicles:
        return

    dest_state = runtime["vehicles"].get(dest_vid)
    if not dest_state or dest_state.get("done"):
        return

    if vehicles[dest_vid]["destination"] is not None:
        return

    if not dest_state.get("waiting_at_airport"):
        return

    if dest_state.get("carrying_final_leg"):
        return

    loaded = _load_specific_boxes(
        sim_state,
        dest_vid,
        job["box_ids"],
        boxes,
        VehicleType.SemiTruck.value.capacity,
    )

    if not loaded:
        return

    dest_state["carrying_final_leg"] = True
    dest_state["route_points"] = job["forward_dest_route"]
    dest_state["next_idx"] = 0

    vehicles = sim_state.get_vehicles()
    if dest_vid in vehicles:
        _send_next_waypoint(sim_state, dest_vid, dest_state, vehicles[dest_vid]["location"])

def step(sim_state):
    """
    Tick 0:
    - build world
    - print summary
    - create test jobs from the planned routes

    Every later tick:
    - keep managed vehicles moving along their stored route
    - unload/load at the right points
    - continue the chosen mode for each route
    """
    global _world

    if sim_state.tick == 0 and not _world:
        _build_world(sim_state)
        _print_world_summary()
        _init_runtime(sim_state)
        return

    runtime = _world.get("runtime")
    if not runtime:
        return

    vehicles = sim_state.get_vehicles()
    boxes = sim_state.get_boxes()

    for vehicle_id, vehicle_state in list(runtime["vehicles"].items()):
        if vehicle_state.get("done"):
            continue

        if vehicle_id not in vehicles:
            continue

        vehicle = vehicles[vehicle_id]
        job = runtime["jobs"][vehicle_state["job_id"]]
        phase = vehicle_state["phase"]

        # still travelling, so wait for next tick
        if vehicle["destination"] is not None:
            continue

        # --------------------------------------------------
        # DIRECT LAND JOBS
        # --------------------------------------------------
        if phase == "direct":
            _unload_deliverable_boxes(sim_state, vehicle_id, vehicle, boxes)

            boxes = sim_state.get_boxes()
            vehicles = sim_state.get_vehicles()
            vehicle = vehicles.get(vehicle_id)
            if not vehicle:
                continue

            if vehicle["cargo"]:
                moved = _send_next_waypoint(sim_state, vehicle_id, vehicle_state, vehicle["location"])
                if not moved:
                    _unload_deliverable_boxes(sim_state, vehicle_id, vehicle, boxes)
                    boxes = sim_state.get_boxes()
                    vehicles = sim_state.get_vehicles()
                    vehicle = vehicles.get(vehicle_id)
                    if vehicle and not vehicle["cargo"]:
                        vehicle_state["done"] = True
                        job["done"] = True
            else:
                vehicle_state["done"] = True
                job["done"] = True

        # --------------------------------------------------
        # AIR COMBO: ORIGIN GROUND LEG
        # --------------------------------------------------
        elif phase == "air_origin_ground":
            if vehicle["cargo"]:
                moved = _send_next_waypoint(sim_state, vehicle_id, vehicle_state, vehicle["location"])
                if moved:
                    continue

                # reached origin airport
                _unload_specific_boxes(sim_state, vehicle_id, list(vehicle["cargo"]))
                vehicle_state["done"] = True
                _try_spawn_and_launch_plane(sim_state, runtime, job)

        # --------------------------------------------------
        # AIR COMBO: PLANE LEG
        # --------------------------------------------------
        elif phase == "air_plane":
            if vehicle["cargo"]:
                # arrived destination airport
                _unload_specific_boxes(sim_state, vehicle_id, list(vehicle["cargo"]))
            vehicle_state["done"] = True
            _try_start_final_ground_leg(sim_state, runtime, job)

        # --------------------------------------------------
        # AIR COMBO: DESTINATION GROUND LEG
        # --------------------------------------------------
        elif phase == "air_dest_ground":
            # empty reposition from hub -> airport
            if not vehicle_state.get("carrying_final_leg"):
                moved = _send_next_waypoint(sim_state, vehicle_id, vehicle_state, vehicle["location"])
                if moved:
                    continue

                # now sitting at destination airport waiting for plane cargo
                vehicle_state["waiting_at_airport"] = True
                _try_start_final_ground_leg(sim_state, runtime, job)

            # loaded final leg airport -> hub
            else:
                _unload_deliverable_boxes(sim_state, vehicle_id, vehicle, boxes)

                boxes = sim_state.get_boxes()
                vehicles = sim_state.get_vehicles()
                vehicle = vehicles.get(vehicle_id)
                if not vehicle:
                    continue

                if vehicle["cargo"]:
                    moved = _send_next_waypoint(sim_state, vehicle_id, vehicle_state, vehicle["location"])
                    if not moved:
                        _unload_deliverable_boxes(sim_state, vehicle_id, vehicle, boxes)
                        boxes = sim_state.get_boxes()
                        vehicles = sim_state.get_vehicles()
                        vehicle = vehicles.get(vehicle_id)
                        if vehicle and not vehicle["cargo"]:
                            vehicle_state["done"] = True
                            job["done"] = True
                else:
                    vehicle_state["done"] = True
                    job["done"] = True
