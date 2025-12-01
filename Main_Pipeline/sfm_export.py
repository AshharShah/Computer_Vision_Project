import json
import numpy as np
from collections import defaultdict
import os

def arr(x):
    return x.tolist() if isinstance(x, np.ndarray) else x

def short_name(path):
    return os.path.basename(path)

def export_sfm_to_json(camera_poses, point_cloud, map_3d_to_keypoint_indices, K, output_prefix=""):
    print("\n--- Exporting SfM Results to JSON ---")
    
    # Check for empty point cloud before proceeding
    if len(point_cloud) == 0:
        print("Skipping JSON export: Point cloud is empty.")
        return

    # Extract Intrinsics from K matrix
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # --- 1. Export Cameras (cameras.json) ---
    cameras_data = {}
    for img_path, (R, t) in camera_poses.items():
        name = short_name(img_path)

        cameras_data[name] = {
            "R": arr(R),
            # Flatten t from (3, 1) to a 3-element list
            "t": arr(t.reshape(-1)), 
            "intrinsics": {
                "fx": float(fx),
                "fy": float(fy),
                "cx": float(cx),
                "cy": float(cy)
            }
        }

    cameras_filename = f"{output_prefix}cameras.json"
    with open(cameras_filename, "w") as f:
        json.dump({"cameras": cameras_data}, f, indent=4)

    print(f"{cameras_filename} written. ({len(cameras_data)} camera poses)")

    # --- Export Points (points.json) ---
    points_list = []
    # Assuming all points are white for simplicity, as color data wasn't tracked
    default_color = [255, 255, 255] 

    for pid, xyz in enumerate(point_cloud):
        # map_3d_to_keypoint_indices[pid] = {full_path: kp_index}
        visibility_paths = list(map_3d_to_keypoint_indices[pid].keys())

        # Convert full paths to simple filenames
        visibility = [short_name(p) for p in visibility_paths]

        point_entry = {
            "id": pid,
            "xyz": arr(xyz),
            "color": default_color,
            "observed_by": visibility
        }
        points_list.append(point_entry)

    points_filename = f"{output_prefix}points.json"
    with open(points_filename, "w") as f:
        json.dump({"points": points_list}, f, indent=4)

    print(f"{points_filename} written. ({len(points_list)} 3D points)")

    # --- Export View Graph (view_graph.json) ---
    edge_weights = defaultdict(int)

    for vis in map_3d_to_keypoint_indices:
        # vis is a dictionary of {full_path: keypoint_id}
        cameras_full = list(vis.keys())
        
        # Convert full paths to simple filenames
        cameras = [short_name(c) for c in cameras_full]

        # Count co-observations (shared 3D points) between every pair of cameras
        for i in range(len(cameras)):
            for j in range(i + 1, len(cameras)):
                pair = tuple(sorted((cameras[i], cameras[j])))
                edge_weights[pair] += 1

    graph_edges = []
    for (cam1, cam2), weight in edge_weights.items():
        graph_edges.append({
            "src": cam1,
            "dst": cam2,
            "weight": int(weight) # Number of co-observed 3D points
        })

    graph_filename = f"{output_prefix}view_graph.json"
    with open(graph_filename, "w") as f:
        json.dump({"graph": graph_edges}, f, indent=4)

    print(f"{graph_filename} written. ({len(graph_edges)} edges)")