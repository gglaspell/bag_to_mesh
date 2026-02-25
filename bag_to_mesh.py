#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bag_to_mesh.py

ROS 2 bag -> registered point cloud -> Poisson mesh.

Key feature added:
- View-ray-tracked normal orientation using per-frame sensor poses:
  1) For each accepted frame, transform points into world.
  2) Compute view rays: ray = sensor_origin - world_point (unit).
  3) Store those rays in the point cloud's .normals channel BEFORE voxel downsampling.
  4) Voxel downsample averages points AND these "normals" (view rays) per voxel.
  5) After filtering, estimate true geometric normals, then flip any normal whose dot(view_ray) < 0.

This eliminates inside-out / flipped-wall Poisson meshes when the sensor trajectory is available.
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# ROS 2 CDR deserialization typestore
TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)

# ---------------------------------------------------------------------------
# PointCloud2 -> Open3D conversion (vectorized, supports FLOAT32/FLOAT64 XYZ)
# ---------------------------------------------------------------------------

# sensor_msgs/PointField datatypes:
# 7 = FLOAT32, 8 = FLOAT64 (common in PointCloud2)
_POINTFIELD_TO_DTYPE = {
    7: np.float32,
    8: np.float64,
}


def convert_ros_pc2_to_o3d(msg):
    """
    Convert ROS 2 sensor_msgs/PointCloud2 to Open3D PointCloud (XYZ only).

    Returns:
        o3d.geometry.PointCloud or None
    """
    try:
        fields = {f.name: (int(f.offset), int(f.datatype)) for f in msg.fields}
        if "x" not in fields or "y" not in fields or "z" not in fields:
            return None

        x_off, x_dt = fields["x"]
        y_off, y_dt = fields["y"]
        z_off, z_dt = fields["z"]

        if x_dt not in _POINTFIELD_TO_DTYPE or y_dt not in _POINTFIELD_TO_DTYPE or z_dt not in _POINTFIELD_TO_DTYPE:
            return None

        # Require consistent dtype across xyz for a clean structured view
        if not (x_dt == y_dt == z_dt):
            return None

        np_dt = _POINTFIELD_TO_DTYPE[x_dt]

        n_points = int(msg.width) * int(msg.height)
        if n_points <= 0:
            return None

        itemsize = int(msg.point_step)
        if itemsize <= 0:
            return None

        # Structured dtype that picks x/y/z at offsets within each point record
        dtype = np.dtype(
            {
                "names": ["x", "y", "z"],
                "formats": [np_dt, np_dt, np_dt],
                "offsets": [x_off, y_off, z_off],
                "itemsize": itemsize,
            }
        )

        arr = np.frombuffer(msg.data, dtype=dtype, count=n_points)
        pts = np.empty((n_points, 3), dtype=np.float64)
        pts[:, 0] = arr["x"]
        pts[:, 1] = arr["y"]
        pts[:, 2] = arr["z"]

        # Filter non-finite
        mask = np.isfinite(pts).all(axis=1)
        pts = pts[mask]
        if pts.shape[0] < 10:
            return None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Odometry -> 4x4 transform
# ---------------------------------------------------------------------------

def get_odom_transform(odom_msg):
    """Extract 4x4 transform from nav_msgs/Odometry pose."""
    try:
        pos = odom_msg.pose.pose.position
        quat = odom_msg.pose.pose.orientation
        t = np.array([pos.x, pos.y, pos.z], dtype=np.float64)
        rot = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot
        T[:3, 3] = t
        return T
    except Exception:
        return None


def get_closest_timestamp(ts, ts_to_value):
    """Find closest timestamp key in a dict to ts."""
    if not ts_to_value:
        return None
    return min(ts_to_value.keys(), key=lambda k: abs(k - ts))


# ---------------------------------------------------------------------------
# Registration helpers (FPFH + RANSAC + ICP) for loop closure
# ---------------------------------------------------------------------------

def compute_fpfh_descriptor(pcd, voxel_size):
    radius_normal = voxel_size * 2.0
    radius_feature = voxel_size * 5.0

    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return fpfh


def ransac_coarse_alignment(source, target, source_fpfh, target_fpfh, voxel_size, ransac_thresh_mult=5.0):
    distance_threshold = voxel_size * float(ransac_thresh_mult)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source,
        target,
        source_fpfh,
        target_fpfh,
        mutual_filter=False,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000, 0.999),
    )
    if result.fitness > 0.1:
        return result.transformation
    return None


def detect_loop_closure(
    current_idx,
    current_pcd,
    current_fpfh,
    historical_pcds,
    historical_fpfhs,
    historical_poses,
    voxel_size,
    search_radius=10.0,
    loop_fitness_thresh=0.3,
    temporal_window=100,
):
    """
    Find loop closures by:
    - only searching older frames (outside temporal window)
    - KD-tree radius search in pose positions
    - RANSAC coarse + ICP refine
    Returns: list of (candidate_idx, transform, fitness)
    """
    if current_idx < temporal_window:
        return []

    search_indices = list(range(0, current_idx - temporal_window))
    if not search_indices:
        return []

    current_pos = historical_poses[current_idx][:3, 3]
    hist_positions = np.array([historical_poses[i][:3, 3] for i in search_indices], dtype=np.float64)
    if hist_positions.shape[0] == 0:
        return []

    pos_pcd = o3d.geometry.PointCloud()
    pos_pcd.points = o3d.utility.Vector3dVector(hist_positions)
    kdtree = o3d.geometry.KDTreeFlann(pos_pcd)
    _, idxs, _ = kdtree.search_radius_vector_3d(current_pos, float(search_radius))
    if not idxs:
        return []

    candidate_indices = [search_indices[i] for i in idxs]

    loop_closures = []
    for cand_idx in candidate_indices:
        cand_fpfh = historical_fpfhs[cand_idx]
        if cand_fpfh is None:
            continue

        coarse = ransac_coarse_alignment(current_pcd, historical_pcds[cand_idx], current_fpfh, cand_fpfh, voxel_size)
        if coarse is None:
            continue

        icp = o3d.pipelines.registration.registration_icp(
            current_pcd,
            historical_pcds[cand_idx],
            voxel_size * 2.0,
            coarse,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20),
        )
        if icp.fitness >= float(loop_fitness_thresh):
            loop_closures.append((cand_idx, icp.transformation, icp.fitness))

    return loop_closures


# ---------------------------------------------------------------------------
# View-ray normal orientation
# ---------------------------------------------------------------------------

def _safe_normalize(v, eps=1e-12):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.clip(n, eps, None)


def attach_view_rays_as_normals(pcd_world, sensor_origin_world):
    """
    Compute unit viewing rays (point -> sensor) and store in pcd_world.normals.
    """
    pts = np.asarray(pcd_world.points, dtype=np.float64)
    if pts.shape[0] == 0:
        return

    ray_dirs = sensor_origin_world.reshape(1, 3) - pts
    ray_dirs = _safe_normalize(ray_dirs, eps=1e-6)
    pcd_world.normals = o3d.utility.Vector3dVector(ray_dirs)


def orient_geometric_normals_with_view_rays(pcd, view_rays):
    """
    Given pcd with estimated normals, flip any normal that points away from view_rays.
    """
    geom = np.asarray(pcd.normals, dtype=np.float64)
    if geom.shape[0] == 0:
        return

    vr = np.asarray(view_rays, dtype=np.float64)
    if vr.shape != geom.shape:
        # If mismatched, do nothing (should not happen if we preserve normals correctly).
        return

    vr = _safe_normalize(vr, eps=1e-6)
    geom = _safe_normalize(geom, eps=1e-12)

    dots = np.sum(geom * vr, axis=1)
    geom[dots < 0.0] *= -1.0
    pcd.normals = o3d.utility.Vector3dVector(geom)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_bag(args):
    bag_path = Path(args.bagpath)
    out_dir = Path(args.outputdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not bag_path.exists():
        sys.exit(f"Error: Bag file not found: {bag_path}")

    topics_to_read = [args.pc_topic]
    if args.odom_topic:
        topics_to_read.append(args.odom_topic)
    if args.imu_topic:
        topics_to_read.append(args.imu_topic)

    pointclouds = []  # list[(ts, o3d PointCloud)]
    odom_data = {}    # dict[ts] = 4x4
    imu_data = {}     # dict[ts] = 3x3 (optional, kept for compatibility)

    print(f"Reading bag: {bag_path}")
    print(f"Output dir: {out_dir}")
    print(f"PointCloud topic: {args.pc_topic}")
    if args.odom_topic:
        print(f"Odometry topic: {args.odom_topic}")
    if args.imu_topic:
        print(f"IMU topic: {args.imu_topic}")
    if args.enable_loop_closure:
        print(f"Loop closure: ENABLED (every {args.loop_closure_search_interval} frames)")
    else:
        print("Loop closure: disabled")

    with AnyReader([bag_path], default_typestore=TYPESTORE) as reader:
        conns = [c for c in reader.connections if c.topic in topics_to_read]
        if not conns:
            sys.exit(f"Error: No messages found for topics: {topics_to_read}")

        for conn, ts, raw in tqdm(reader.messages(connections=conns), desc="Reading messages"):
            try:
                msg = reader.deserialize(raw, conn.msgtype)

                if conn.topic == args.pc_topic:
                    pcd = convert_ros_pc2_to_o3d(msg)
                    if pcd is not None and len(pcd.points) >= 100:
                        pointclouds.append((ts, pcd))

                elif args.odom_topic and conn.topic == args.odom_topic:
                    T = get_odom_transform(msg)
                    if T is not None:
                        odom_data[ts] = T

                elif args.imu_topic and conn.topic == args.imu_topic:
                    # Optional: store IMU orientation (not used directly in this pipeline)
                    try:
                        quat = msg.orientation
                        imu_data[ts] = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()
                    except Exception:
                        pass

            except Exception:
                continue

    if not pointclouds:
        sys.exit("Error: No valid point clouds were extracted.")

    print(f"Extracted {len(pointclouds)} point clouds")
    if args.odom_topic:
        print(f"Extracted {len(odom_data)} odometry messages")
    if args.imu_topic:
        print(f"Extracted {len(imu_data)} IMU messages")

    # -----------------------------------------------------------------------
    # 1) Pairwise registration + pose graph
    # -----------------------------------------------------------------------
    posegraph = o3d.pipelines.registration.PoseGraph()

    current_transform = np.eye(4, dtype=np.float64)
    posegraph.nodes.append(o3d.pipelines.registration.PoseGraphNode(current_transform.copy()))

    # First frame
    _, source_raw = pointclouds[0]
    source = source_raw.voxel_down_sample(args.voxel_size)
    source.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * 2.0, max_nn=30)
    )

    source_fpfh = None
    if args.enable_loop_closure and (0 % args.loop_closure_search_interval == 0):
        source_fpfh = compute_fpfh_descriptor(source, args.voxel_size)

    accumulated_pcds = [source]
    accumulated_fpfhs = [source_fpfh]
    accumulated_poses = [current_transform.copy()]

    previous_odom_T = None
    if args.odom_topic:
        closest_ts = get_closest_timestamp(pointclouds[0][0], odom_data)
        if closest_ts is not None:
            previous_odom_T = odom_data[closest_ts]

    # Map posegraph node index -> original pointcloud index
    successful_pc_indices = [0]
    loop_closures_found = 0

    print("Registering point clouds...")
    for i in tqdm(range(1, len(pointclouds)), desc="Registering"):
        ts, target_raw = pointclouds[i]

        target = target_raw.voxel_down_sample(args.voxel_size)
        target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * 2.0, max_nn=30)
        )

        initial_guess = np.eye(4, dtype=np.float64)
        if args.odom_topic:
            closest_ts = get_closest_timestamp(ts, odom_data)
            if closest_ts is not None:
                current_odom_T = odom_data[closest_ts]
                if previous_odom_T is not None:
                    # relative motion in odom frame for ICP initial guess
                    initial_guess = np.linalg.inv(previous_odom_T) @ current_odom_T
                previous_odom_T = current_odom_T

        try:
            reg = o3d.pipelines.registration.registration_icp(
                source,
                target,
                args.icp_dist_thresh,
                initial_guess,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
            )
        except Exception:
            continue

        if reg.fitness < args.icp_fitness_thresh:
            continue

        # Update running transform and add node/edge
        current_transform = reg.transformation @ current_transform

        posegraph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(current_transform))
        )

        information = np.eye(6, dtype=np.float64) * float(max(reg.fitness, 1e-6))
        posegraph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                len(posegraph.nodes) - 2,
                len(posegraph.nodes) - 1,
                reg.transformation,
                information,
                uncertain=False,
            )
        )

        accumulated_pcds.append(target)
        accumulated_poses.append(current_transform.copy())

        # FPFH only on keyframes (interval)
        target_fpfh = None
        do_lc = args.enable_loop_closure and (i % args.loop_closure_search_interval == 0)
        if do_lc:
            target_fpfh = compute_fpfh_descriptor(target, args.voxel_size)
        accumulated_fpfhs.append(target_fpfh)

        # Loop closure detection
        if do_lc:
            lcs = detect_loop_closure(
                current_idx=len(accumulated_pcds) - 1,
                current_pcd=target,
                current_fpfh=target_fpfh,
                historical_pcds=accumulated_pcds,
                historical_fpfhs=accumulated_fpfhs,
                historical_poses=accumulated_poses,
                voxel_size=args.voxel_size,
                search_radius=args.loop_closure_radius,
                loop_fitness_thresh=args.loop_closure_fitness_thresh,
                temporal_window=100,
            )
            for cand_idx, lc_T, lc_fit in lcs:
                lc_info = np.eye(6, dtype=np.float64) * float(max(lc_fit, 1e-6) * 100.0)
                posegraph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        cand_idx,
                        len(posegraph.nodes) - 1,
                        lc_T,
                        lc_info,
                        uncertain=True,
                    )
                )
                loop_closures_found += 1

        successful_pc_indices.append(i)

        # advance
        source = target

    if len(posegraph.nodes) < 2:
        sys.exit("Error: Registration failed (too few successful registrations).")

    if args.enable_loop_closure:
        print(f"Loop closures detected: {loop_closures_found}")

    # -----------------------------------------------------------------------
    # 2) Global optimization
    # -----------------------------------------------------------------------
    print("Optimizing pose graph...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=args.icp_dist_thresh,
        edge_prune_threshold=0.25,
        reference_node=0,
    )
    try:
        o3d.pipelines.registration.global_optimization(
            posegraph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option,
        )
    except Exception as e:
        print(f"Warning: Global optimization failed: {e}")

    # -----------------------------------------------------------------------
    # 3) Merge clouds AND hijack normals with view rays BEFORE voxel downsample
    # -----------------------------------------------------------------------
    print("Merging clouds (tracking viewing rays in normals)...")
    pcd_combined = o3d.geometry.PointCloud()

    for node_idx, pc_idx in tqdm(list(enumerate(successful_pc_indices)), desc="Merging", total=len(successful_pc_indices)):
        _, pcd_raw = pointclouds[pc_idx]
        pose = np.asarray(posegraph.nodes[node_idx].pose, dtype=np.float64)

        # Clone to avoid mutating stored raw clouds using deepcopy
        pcd_world = copy.deepcopy(pcd_raw)
        pcd_world.transform(pose)

        # Store view rays in normals channel
        sensor_origin = pose[:3, 3].copy()
        attach_view_rays_as_normals(pcd_world, sensor_origin)

        pcd_combined += pcd_world

    if len(pcd_combined.points) == 0:
        sys.exit("Error: Combined point cloud is empty.")

    # Optional: floor leveling (apply same rotation to points AND view-ray normals)
    if args.level_floor:
        print("Attempting floor leveling...")
        try:
            # Use a downsample for robust/fast plane fit
            pcd_tmp = pcd_combined.voxel_down_sample(args.voxel_size * 2.0)
            plane_model, inliers = pcd_tmp.segment_plane(
                distance_threshold=args.voxel_size * 2.0,
                ransac_n=3,
                num_iterations=1000,
            )
            a, b, c, _d = plane_model
            n = np.array([a, b, c], dtype=np.float64)
            n = n / (np.linalg.norm(n) + 1e-12)

            target_n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if np.dot(n, target_n) < 0:
                n = -n

            v = np.cross(n, target_n)
            s = np.linalg.norm(v)
            if s > 1e-12:
                cang = float(np.dot(n, target_n))
                vx = np.array(
                    [[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]],
                    dtype=np.float64,
                )
                R3 = np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - cang) / (s * s))

                pts = np.asarray(pcd_combined.points, dtype=np.float64)
                pts = pts @ R3.T
                pcd_combined.points = o3d.utility.Vector3dVector(pts)

                if pcd_combined.has_normals():
                    nr = np.asarray(pcd_combined.normals, dtype=np.float64)
                    nr = nr @ R3.T
                    pcd_combined.normals = o3d.utility.Vector3dVector(nr)

                print("Floor leveling applied.")
            else:
                print("Map is already level.")
        except Exception as e:
            print(f"Warning: Floor leveling failed: {e}")

    # -----------------------------------------------------------------------
    # 4) Cleaning (Voxel, ROR, SOR, DBSCAN) while preserving view rays
    # -----------------------------------------------------------------------
    print("Voxel downsampling (this averages viewing rays)...")
    pcd_vox = pcd_combined.voxel_down_sample(args.voxel_size)
    pcd_clean = pcd_vox

    # 4.1 Radius Outlier Removal
    print("Radius outlier removal (cleaning wispy noise)...")
    try:
        pcd_tmp, _ = pcd_clean.remove_radius_outlier(nb_points=12, radius=args.voxel_size * 3.0)
        if len(pcd_tmp.points) > 0:
            pcd_clean = pcd_tmp
        else:
            print("Warning: ROR produced empty cloud; skipping this filter.")
    except Exception as e:
        print(f"Warning: ROR failed ({e}); skipping ROR.")

    # 4.2 Statistical Outlier Removal
    print("Statistical outlier removal (cleaning distant noise)...")
    try:
        pcd_tmp, _ = pcd_clean.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        if len(pcd_tmp.points) > 0:
            pcd_clean = pcd_tmp
        else:
            print("Warning: SOR produced empty cloud; skipping this filter.")
    except Exception as e:
        print(f"Warning: SOR failed ({e}); skipping SOR.")

    # 4.3 DBSCAN Clustering (Removes floating artifacts/components)
    print("DBSCAN clustering to retain only the main continuous structure...")
    try:
        labels = np.array(pcd_clean.cluster_dbscan(eps=args.voxel_size * 4.0, min_points=30, print_progress=False))
        if len(labels) > 0 and labels.max() >= 0:
            largest_cluster_idx = np.bincount(labels[labels >= 0]).argmax()
            pcd_tmp = pcd_clean.select_by_index(np.where(labels == largest_cluster_idx)[0])
            if len(pcd_tmp.points) > 0:
                pcd_clean = pcd_tmp
            else:
                print("Warning: DBSCAN isolated 0 points; skipping clustering filter.")
    except Exception as e:
        print(f"Warning: DBSCAN failed ({e}); skipping DBSCAN.")

    # Save final point cloud (with view rays still in normals) for debugging
    ply_path = out_dir / f"{bag_path.stem}_cloud.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd_clean)
    print(f"Saved point cloud: {ply_path}")

    # Extract averaged viewing rays BEFORE overwriting normals
    viewing_rays = None
    if pcd_clean.has_normals():
        viewing_rays = np.asarray(pcd_clean.normals, dtype=np.float64).copy()

    print("Estimating geometric normals for meshing...")
    pcd_clean.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel_size * 3.0, max_nn=30)
    )
    pcd_clean.normalize_normals()

    if viewing_rays is not None and viewing_rays.shape[0] == len(pcd_clean.points):
        print("Orienting geometric normals using historical viewing rays...")
        orient_geometric_normals_with_view_rays(pcd_clean, viewing_rays)
    else:
        print("Warning: Viewing rays unavailable or mismatched; normals remain unoriented.")

    # -----------------------------------------------------------------------
    # 5) Poisson reconstruction + cleanup
    # -----------------------------------------------------------------------
    print("Reconstructing mesh (Poisson)...")
    obj_path = None
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_clean, depth=9
        )

        print(f"Initial mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

        densities = np.asarray(densities, dtype=np.float64)
        if densities.size > 0:
            thresh = np.quantile(densities, 0.05)
            mask = densities < thresh
            mesh.remove_vertices_by_mask(mask)
            print(f"After density trim: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

        # Cleanup
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()

        obj_path = out_dir / f"{bag_path.stem}_mesh.obj"
        o3d.io.write_triangle_mesh(str(obj_path), mesh)
        print(f"Saved mesh: {obj_path}")

    except Exception as e:
        print(f"Warning: Mesh generation failed: {e}")

    print("Done.")
    print(f"Point cloud: {ply_path}")
    print(f"Mesh: {obj_path if obj_path is not None else 'N/A'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert ROS 2 bag files with PointCloud2 data to a 3D mesh (Poisson)."
    )
    parser.add_argument("bagpath", help="Path to the ROS 2 bag file.")
    parser.add_argument("outputdir", help="Directory to save output files.")

    parser.add_argument("--pc_topic", default="points", help="PointCloud2 topic name.")
    parser.add_argument("--odom_topic", default=None, help="Odometry topic for initial guess (nav_msgs/Odometry).")
    parser.add_argument("--imu_topic", default=None, help="IMU topic (optional; currently unused).")

    parser.add_argument("--voxel_size", type=float, default=0.05, help="Voxel size for downsampling (m).")
    parser.add_argument("--icp_dist_thresh", type=float, default=0.2, help="ICP max correspondence distance (m).")
    parser.add_argument("--icp_fitness_thresh", type=float, default=0.6, help="Minimum ICP fitness to accept a frame (0-1).")

    parser.add_argument("--enable_loop_closure", action="store_true", default=False, help="Enable loop closure detection.")
    parser.add_argument("--loop_closure_radius", type=float, default=10.0, help="Max distance to search for loop closures (m).")
    parser.add_argument("--loop_closure_fitness_thresh", type=float, default=0.3, help="Minimum ICP fitness for loop closure.")
    parser.add_argument("--loop_closure_search_interval", type=int, default=10, help="Frequency of loop closure search (every N frames).")

    parser.add_argument("--level_floor", action="store_true", help="Attempt to level the final map with the floor plane.")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    process_bag(args)


if __name__ == "__main__":
    main()
