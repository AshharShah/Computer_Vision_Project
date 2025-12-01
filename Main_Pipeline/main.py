import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

from sfm_utils import (
    load_images_and_paths, compute_features, compute_matches, 
    get_aligned_points, visualize_matches, create_intrinsic_matrix
)
from sfm_solver import (
    find_pose_and_filter, triangulate_points, run_global_bundle_adjustment_optimized
)
from sfm_export import (
    export_sfm_to_json
)

output_dir = r"../Tehzeeb_v2/resized_images" 
min_triangulation_angle = 3.0 # Minimum angle for valid triangulation (in degrees)

def run_sfm_pipeline():
    # --- Global Data Structures ---
    global features, descriptors, camera_poses, point_cloud, map_3d_to_keypoint_indices
    features = {}                     # Stores keypoints (kp) for each image file
    descriptors = {}                  # Stores descriptors (desc) for each image file
    camera_poses = {}                 # Stores (R, t) for each image file
    point_cloud = np.empty((0, 3), dtype=np.float64) # The 3D map (will be updated)
    map_3d_to_keypoint_indices = []   # Map: 3D point index -> {img_path: kp_idx}

    resized_paths, images = load_images_and_paths(output_dir)

    print(f"Total images loaded: {len(images)}")

    if not images:
        print("No images found. Check your 'output_dir' path.")
        return
    
    img_sample = images[0]
    K = create_intrinsic_matrix(img_sample)
    print(f"\nIntrinsic Matrix K:\n{K}")

    print("\n--- Computing Features ---")
    for i, img in enumerate(images):
        img_path = resized_paths[i]
        kp, desc = compute_features(img)
        print(f"Detecting Features For: {img_path} - Found {len(kp)}")
        
        features[img_path] = kp
        descriptors[img_path] = desc

    if len(images) < 2:
        print("Need at least two images for initial reconstruction.")
        return

    img1_file, img2_file = resized_paths[0], resized_paths[1]
    kp1, desc1 = features[img1_file], descriptors[img1_file]
    kp2, desc2 = features[img2_file], descriptors[img2_file]

    matches = compute_matches(desc1, desc2)
    print(f"\nFound {len(matches)} matches between Image 1 & Image 2")
    
    if not matches:
        print("Not enough matches for initial reconstruction.")
        return

    # Get points and indices aligned by the matches
    pts1, pts2, idx1, idx2 = get_aligned_points(kp1, kp2, matches)

    # Estimate Essential Matrix and recover Pose (R, t) using RANSAC
    R, t, pts1_inliers, pts2_inliers, mask_inliers = find_pose_and_filter(pts1, pts2, K)

    idx1_inliers = idx1[mask_inliers]
    idx2_inliers = idx2[mask_inliers]

    # Set initial camera poses (Camera 1 at World Origin)
    R1, t1 = np.eye(3), np.zeros((3,1))
    R2, t2 = R, t
    camera_poses = {img1_file: (R1, t1), img2_file: (R2, t2)}

    # Triangulate 3D points
    new_pts3d_raw, angle_mask = triangulate_points(
        K, R1, t1, R2, t2, pts1_inliers, pts2_inliers, min_angle_deg=min_triangulation_angle
    )

    # Filter indices for the successfully triangulated points
    idx1_initial = idx1_inliers[angle_mask]
    idx2_initial = idx2_inliers[angle_mask]

    point_cloud = new_pts3d_raw

    # Update map
    for i in range(len(point_cloud)):
        map_3d_to_keypoint_indices.append({
            img1_file: idx1_initial[i],
            img2_file: idx2_initial[i]
        })
    print(f"Initial triangulation completed. Total 3D points: {len(point_cloud)}")

    # Sequential (Incremental) Structure from Motion
    for img_idx in range(2, len(images)):
        
        print(f"\n--- Processing Image {img_idx+1}/{len(images)}: {resized_paths[img_idx]} ---")

        curr_img_path = resized_paths[img_idx]
        prev_img_path = resized_paths[img_idx - 1] # Match against previous image

        prev_kp, prev_desc = features[prev_img_path], descriptors[prev_img_path]
        curr_kp, curr_desc = features[curr_img_path], descriptors[curr_img_path]

        # Match features between the previous and the current image
        matches_prev_curr = compute_matches(prev_desc, curr_desc)
        print(f"Matches Found Between Image {img_idx} & Image {img_idx+1}: {len(matches_prev_curr)}")

        if len(matches_prev_curr) < 20:
            print(f"Skipping {curr_img_path}, insufficient matches.")
            continue

        (
            prev_matched_pts, curr_matched_pts, 
            prev_matched_idx, curr_matched_idx,
        ) = get_aligned_points(prev_kp, curr_kp, matches_prev_curr)

        # Build 2D-3D correspondences for PnP
        pnp_points_2d = []
        pnp_points_3d = []
        kp_to_3d_index = {}
        matches_for_triangulation = []

        # Map 3D points to previous image's keypoint indices
        for point3d_idx, kp_map in enumerate(map_3d_to_keypoint_indices):
            if prev_img_path in kp_map:
                prev_kp_idx = kp_map[prev_img_path]
                kp_to_3d_index[prev_kp_idx] = point3d_idx

        # Separate matches for PnP (existing 3D points) and Triangulation (new points)
        for m_idx in range(len(matches_prev_curr)):
            kp_idx_prev = prev_matched_idx[m_idx]

            if kp_idx_prev in kp_to_3d_index:
                point3d_idx = kp_to_3d_index[kp_idx_prev]
                pnp_points_2d.append(curr_matched_pts[m_idx])
                pnp_points_3d.append(point_cloud[point3d_idx])
                
                # Update map: link this existing 3D point to the current image's keypoint
                map_3d_to_keypoint_indices[point3d_idx][curr_img_path] = curr_matched_idx[m_idx]
            else:
                matches_for_triangulation.append(m_idx)

        pnp_points_2d = np.array(pnp_points_2d, dtype=np.float64)
        pnp_points_3d = np.array(pnp_points_3d, dtype=np.float64)

        print(f"Image {img_idx+1}: Found {len(pnp_points_3d)} 2D-3D correspondences for PnP.")

        # Solve PnP to Estimate Current Camera Pose
        if len(pnp_points_3d) >= 6:
            # Use solvePnPRansac to robustly estimate R, t
            _, rvec, tvec, inliers = cv2.solvePnPRansac(
                pnp_points_3d, pnp_points_2d, K, None,
                flags=cv2.SOLVEPNP_ITERATIVE,
                confidence=0.85,
                iterationsCount=5_000_000,
                reprojectionError=15.0 # Error tolerance
            )

            R_curr, _ = cv2.Rodrigues(rvec)
            t_curr = tvec
            camera_poses[curr_img_path] = (R_curr, t_curr)
            print(f"PnP successful for {curr_img_path}. Inliers: {len(inliers)}")
            
        else:
            print(f"Skipping PnP for {curr_img_path}, not enough correspondences (< 6).")
            continue

        # Triangulate New Points
        R_prev, t_prev = camera_poses[prev_img_path]

        prev_pts_new = prev_matched_pts[matches_for_triangulation]
        curr_pts_new = curr_matched_pts[matches_for_triangulation]
        prev_kp_idx_new = prev_matched_idx[matches_for_triangulation]
        curr_kp_idx_new = curr_matched_idx[matches_for_triangulation]
        
        if len(prev_pts_new) > 0:
            new_pts3d_raw, valid_mask = triangulate_points(
                K, R_prev, t_prev, R_curr, t_curr,
                prev_pts_new, curr_pts_new,
                min_angle_deg=min_triangulation_angle
            )

            valid_prev_idx = prev_kp_idx_new[valid_mask]
            valid_curr_idx = curr_kp_idx_new[valid_mask]
            
            # Insert new points into global 3D map
            if len(new_pts3d_raw) > 0:
                point_cloud = np.vstack((point_cloud, new_pts3d_raw))
                for k in range(len(new_pts3d_raw)):
                    map_3d_to_keypoint_indices.append({
                        prev_img_path: valid_prev_idx[k],
                        curr_img_path: valid_curr_idx[k]
                    })

                print(f"Added {len(new_pts3d_raw)} new 3D points. Total map size: {len(point_cloud)}")
            else:
                 print("No new points passed angle + cheirality filters.")

        # Global Bundle Adjustment (Run periodically)
        # Run GBA every 2 images
        if(img_idx % 2 == 0):
            print("\nRunning Global Bundle Adjustment...")
            camera_poses, point_cloud = run_global_bundle_adjustment_optimized(
                camera_poses,
                point_cloud,
                map_3d_to_keypoint_indices,
                features,
                K
            )
            print("Finished Global Bundle Adjustment.\n")

    print(f"\n*** Incremental SfM Complete. Final Point Cloud Size: {len(point_cloud)} ***")
    
    # Export To JSON for visualization
    export_sfm_to_json(
        camera_poses, 
        point_cloud, 
        map_3d_to_keypoint_indices, 
        K
    )

if __name__ == "__main__":
    run_sfm_pipeline()