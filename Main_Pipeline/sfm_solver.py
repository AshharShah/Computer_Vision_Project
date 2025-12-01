import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix, csr_matrix
from numba import njit, prange

@njit
def _rodrigues_batch_numba(rvecs):
    n = rvecs.shape[0]
    R_out = np.empty((n, 3, 3), dtype=np.float32)
    
    for i in range(n):
        rx, ry, rz = rvecs[i]
        theta2 = rx*rx + ry*ry + rz*rz

        if theta2 > 1e-12:
            theta = np.sqrt(theta2)
            kx, ky, kz = rx/theta, ry/theta, rz/theta
            c, s = np.cos(theta), np.sin(theta)
            v = 1.0 - c

            R_out[i, 0, 0] = kx*kx*v + c
            R_out[i, 0, 1] = kx*ky*v - kz*s
            R_out[i, 0, 2] = kx*kz*v + ky*s

            R_out[i, 1, 0] = ky*kx*v + kz*s
            R_out[i, 1, 1] = ky*ky*v + c
            R_out[i, 1, 2] = ky*kz*v - kx*s

            R_out[i, 2, 0] = kz*kx*v - ky*s
            R_out[i, 2, 1] = kz*ky*v + kx*s
            R_out[i, 2, 2] = kz*kz*v + c
        else:
            # Near-zero rotation = identity
            R_out[i] = np.eye(3, dtype=np.float32)

    return R_out

@njit(parallel=True)
def _project_numba(R_all, t_all, pts, cam_indices, fx, fy, cx, cy):
    n = pts.shape[0]
    out = np.empty((n, 2), dtype=np.float32)
    
    for i in prange(n):
        c = cam_indices[i]
        R = R_all[c]
        t0, t1, t2 = t_all[c]
        X, Y, Z = pts[i]

        # Apply extrinsic transformation [R | t] * X
        xc = R[0,0]*X + R[0,1]*Y + R[0,2]*Z + t0
        yc = R[1,0]*X + R[1,1]*Y + R[1,2]*Z + t1
        zc = R[2,0]*X + R[2,1]*Y + R[2,2]*Z + t2
        
        # Pinhole camera model (projection)
        if zc == 0:
            zc = 1e-8
        
        u = fx * (xc / zc) + cx
        v = fy * (yc / zc) + cy
        out[i, 0], out[i, 1] = u, v
    return out

def find_pose_and_filter(pts1, pts2, K):
    # Convert points to the format required by cv2 functions
    pts1_cv = pts1.astype(np.float64).reshape(-1, 1, 2)
    pts2_cv = pts2.astype(np.float64).reshape(-1, 1, 2)

    # Fetch the Essential Matrix (E)
    E, mask = cv2.findEssentialMat(pts1_cv, pts2_cv, K, method=cv2.RANSAC, prob=0.99, threshold=2.0)
    
    # Fetch the Rotation (R) & Translation (t)
    _, R, t, mask_pose = cv2.recoverPose(E, pts1_cv, pts2_cv, K, mask=mask)

    # Flatten the mask_pose into a 1D boolean array for inliers
    mask_inliers = mask_pose.ravel() == 1

    # Filter points to keep only inliers
    pts1_inliers = pts1[mask_inliers]
    pts2_inliers = pts2[mask_inliers]
    
    # Return (R, t) for the second camera, assuming the first is (I, 0)
    return R, t, pts1_inliers, pts2_inliers, mask_inliers

def triangulate_points(K, R1, t1, R2, t2, pts1, pts2, min_angle_deg):
    # Normalize the Image Points (Convert to Camera Coordinates)
    # The normalization process is simplified by using cv2.undistortPoints
    # which is often used but the original logic using inv(K) is kept for consistency:
    pts1_h = cv2.convertPointsToHomogeneous(pts1)[:, 0, :]
    pts2_h = cv2.convertPointsToHomogeneous(pts2)[:, 0, :]
    
    K_inv = np.linalg.inv(K)
    # Normalize points using K_inv
    pts1_norm = cv2.convertPointsFromHomogeneous((K_inv @ pts1_h.T).T)[:, 0, :]
    pts2_norm = cv2.convertPointsFromHomogeneous((K_inv @ pts2_h.T).T)[:, 0, :]

    # Build the Project Matrices P=[R∣t]
    P1 = np.hstack((R1, t1))
    P2 = np.hstack((R2, t2))
    
    # Perform Triangulation
    pts4d = cv2.triangulatePoints(P1, P2, pts1_norm.T, pts2_norm.T)
    pts3d = cv2.convertPointsFromHomogeneous(pts4d.T)[:, 0, :]

    # Angle Check
    C1 = -R1.T @ t1
    C2 = -R2.T @ t2
    ray1 = pts3d - C1.T
    ray2 = pts3d - C2.T
    
    ray1_norm = ray1 / np.linalg.norm(ray1, axis=1)[:, np.newaxis]
    ray2_norm = ray2 / np.linalg.norm(ray2, axis=1)[:, np.newaxis]

    cos_angle = np.sum(ray1_norm * ray2_norm, axis=1)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_degrees = np.degrees(np.arccos(cos_angle))

    angle_mask = angle_degrees > min_angle_deg
    
    # Cheirality Check (Depth must be positive for both cameras)
    # Transform points to each camera's coordinate system and check Z > 0
    pts3d_in_prev = R1 @ pts3d.T + t1
    pts3d_in_curr = R2 @ pts3d.T + t2
    cheirality_mask = (pts3d_in_prev[2, :] > 0) & (pts3d_in_curr[2, :] > 0)
    
    valid_mask = angle_mask & cheirality_mask

    return pts3d[valid_mask], valid_mask

# --- Bundle Adjustment Functions (From original notebook) ---

def build_observations(camera_poses, point_cloud, kp_mapping, features):
    cam_list = list(camera_poses.keys())
    cam_to_idx = {name: i for i, name in enumerate(cam_list)}
    pts3d = np.asarray(point_cloud, dtype=np.float32)
    
    cam_idx_list = []
    pt_idx_list = []
    xy_list = []
    
    for pt_idx, kp_map in enumerate(kp_mapping):
        for img, kp_id in kp_map.items():
            if img not in cam_to_idx:
                continue
            cam_id = cam_to_idx[img]
            kp = features[img][kp_id]
            x, y = kp.pt
            
            cam_idx_list.append(cam_id)
            pt_idx_list.append(pt_idx)
            xy_list.append([x, y])
            
    if not cam_idx_list:
        return None

    return (
        cam_list,
        cam_to_idx,
        pts3d,
        np.array(cam_idx_list, dtype=np.int32),
        np.array(pt_idx_list, dtype=np.int32),
        np.array(xy_list, dtype=np.float32),
    )

def pack_camera_params(rvecs, tvecs, opt_cam_indices):
    params = []
    for i in opt_cam_indices:
        params.append(np.asarray(rvecs[i], dtype=np.float32))
        params.append(np.asarray(tvecs[i], dtype=np.float32))
    return np.hstack(params).astype(np.float32) if params else np.zeros(0, np.float32)

def build_jac_sparsity(n_cams, n_points, cam_ids, pt_ids, opt_cam_map):
    obs_n = cam_ids.shape[0]
    n_opt_cams = sum(v >= 0 for v in opt_cam_map.values())
    total_vars = 6*n_opt_cams + 3*n_points

    J = lil_matrix((2*obs_n, total_vars), dtype=np.int8)

    for o in range(obs_n):
        cam = cam_ids[o]
        pt = pt_ids[o]

        # Camera block
        cam_order = opt_cam_map.get(cam, -1)
        if cam_order >= 0:
            off = cam_order * 6
            J[2*o, off:off+6] = 1
            J[2*o+1, off:off+6] = 1

        # Point block
        pt_off = 6*n_opt_cams + 3*pt
        J[2*o, pt_off:pt_off+3] = 1
        J[2*o+1, pt_off:pt_off+3] = 1
        
    return csr_matrix(J)

def gba_residuals_numba(
    x, n_cams, n_points, opt_cam_indices, opt_cam_map,
    fixed_rvecs, fixed_tvecs,
    cam_ids, pt_ids, xy_obs, K
):
    n_opt = len(opt_cam_indices)
    cam_block = 6*n_opt
    x_cam = x[:cam_block].astype(np.float32)
    x_pts = x[cam_block:].astype(np.float32).reshape((n_points, 3))

    rvecs = [r.copy() for r in fixed_rvecs]
    tvecs = [t.copy() for t in fixed_tvecs]

    off = 0
    for cam in opt_cam_indices:
        rvecs[cam] = x_cam[off:off+3]
        tvecs[cam] = x_cam[off+3:off+6]
        off += 6
    
    r_arr = np.stack(rvecs).astype(np.float32)
    t_arr = np.stack(tvecs).astype(np.float32)
    
    R_all = _rodrigues_batch_numba(r_arr)

    pts_obs = x_pts[pt_ids]
    
    fx, fy = np.float32(K[0,0]), np.float32(K[1,1])
    cx, cy = np.float32(K[0,2]), np.float32(K[1,2])
    
    pred = _project_numba(R_all, t_arr, pts_obs, cam_ids, fx, fy, cx, cy)
    
    # Compute the difference between predicted and observed 2D points (loss)
    return (pred - xy_obs).ravel().astype(np.float32)


def run_global_bundle_adjustment_optimized(
    camera_poses, point_cloud, kp_map, features, K,
    fix_first_camera=True, max_nfev=200
):
    if camera_poses is None or point_cloud is None:
        print("[Bundle Adjustment] Invalid input (No Camera Poses / Point Cloud), skipping...")
        return camera_poses, point_cloud

    obs = build_observations(camera_poses, point_cloud, kp_map, features)
    if obs is None:
        print("[Bundle Adjustment] No observations found, skipping...")
        return camera_poses, point_cloud
        
    cam_list, cam_to_idx, pts3d, cam_ids, pt_ids, xy = obs
    n_cams, n_points = len(cam_list), pts3d.shape[0]

    if xy.shape[0] == 0:
        print("[Bundle Adjustment] No matches found, skipping...")
        return camera_poses, point_cloud

    # Convert R to rvec and collect tvec
    r_init, t_init = [], []
    for name in cam_list:
        R, t = camera_poses[name]
        rv, _ = cv2.Rodrigues(R)
        r_init.append(rv.astype(np.float32).ravel())
        t_init.append(t.astype(np.float32).ravel())
        
    fixed_r, fixed_t = [r.copy() for r in r_init], [t.copy() for t in t_init]

    # Select which cameras to optimize (fix the first one)
    opt_cam_indices = list(range(n_cams))
    if fix_first_camera and n_cams > 0:
        opt_cam_indices = [i for i in range(n_cams) if i != 0]
    
    opt_cam_map = {}
    order = 0
    for i in range(n_cams):
        if i in opt_cam_indices:
            opt_cam_map[i] = order
            order += 1
        else:
            opt_cam_map[i] = -1

    # Pack initial parameters (camera + points)
    x0_cam = pack_camera_params(r_init, t_init, opt_cam_indices)
    x0_pts = pts3d.ravel().astype(np.float32)
    x0 = np.hstack([x0_cam, x0_pts]).astype(np.float32)

    # Build Jacobian sparsity
    J_pattern = build_jac_sparsity(n_cams, n_points, cam_ids, pt_ids, opt_cam_map)

    print(f"Bundle Adjustment: {n_cams} cams, {n_points} pts, {xy.shape[0]} obs")
    
    # Define residual function
    def residual_fn(x):
        return gba_residuals_numba(
            x, n_cams, n_points, opt_cam_indices, opt_cam_map,
            fixed_r, fixed_t, cam_ids, pt_ids, xy, K
        )

    # Run the optimizer
    result = least_squares(
        residual_fn, x0,
        method="trf",
        jac="2-point",
        jac_sparsity=J_pattern,
        max_nfev=max_nfev,
        loss="huber",
        ftol=1e-6,
        xtol=1e-6,
        verbose=2
    )

    print("Bundle Adjustment finished:", result.success, "|", result.message)

    # Unpack the optimized parameters
    x_opt = result.x.astype(np.float32)
    cam_block = 6*len(opt_cam_indices)
    cam_opt = x_opt[:cam_block]
    pts_opt = x_opt[cam_block:].reshape((n_points, 3))

    # Update the camera poses
    r_new, t_new = fixed_r.copy(), fixed_t.copy()

    off = 0
    for cam in opt_cam_indices:
        r_new[cam] = cam_opt[off:off+3]
        t_new[cam] = cam_opt[off+3:off+6]
        off += 6
        
    for i, name in enumerate(cam_list):
        R, _ = cv2.Rodrigues(r_new[i].astype(np.float64))
        t = t_new[i].reshape(3,1).astype(np.float64)
        camera_poses[name] = (R, t)

    # Update the point cloud
    point_cloud[:] = pts_opt.astype(np.float32)

    return camera_poses, point_cloud