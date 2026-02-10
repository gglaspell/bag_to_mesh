#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
bag_to_mesh.py with Full Loop Closure Detection (Optimized & Corrected)

ROS 2 bag to 3D mesh converter with:
- Frame-to-map loop closure detection using FPFH descriptors
- RANSAC-based coarse alignment
- ICP refinement for loop closure constraints
- Improved pose graph optimization

OPTIMIZATIONS & CORRECTIONS:
- FIXED: Loop closure constraints are now correctly weighted by fitness.
- FIXED: Fragile node indexing replaced with a robust counter.
- OPTIMIZED: FPFH features are only computed for keyframes, improving performance.
- OPTIMIZED: Periodic loop closure detection (not every frame).
- OPTIMIZED: KD-Tree for fast loop closure candidate search.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# Get typestore for ROS 2 CDR deserialization
typestore = get_typestore(Stores.ROS2_HUMBLE)

# ---------------------------------------------------------------------------
# PointCloud2 → Open3D conversion
# ---------------------------------------------------------------------------

def convert_ros_pc2_to_o3d(msg):
    """
    Converts a ROS 2 sensor_msgs/PointCloud2 message to an Open3D PointCloud.
    Handles different field types and extracts XYZ coordinates.
    """
    field_map = {field.name: (field.offset, field.datatype) for field in msg.fields}

    if 'x' not in field_map or 'y' not in field_map or 'z' not in field_map:
        return None

    points = []
    for i in range(msg.width * msg.height):
        start_byte = i * msg.point_step
        x_offset, x_dtype = field_map['x']
        y_offset, y_dtype = field_map['y']
        z_offset, z_dtype = field_map['z']

        try:
            # FLOAT32
            if x_dtype == 7 and y_dtype == 7 and z_dtype == 7:
                x = np.frombuffer(
                    msg.data[start_byte + x_offset:start_byte + x_offset + 4],
                    dtype=np.float32
                )[0]
                y = np.frombuffer(
                    msg.data[start_byte + y_offset:start_byte + y_offset + 4],
                    dtype=np.float32
                )[0]
                z = np.frombuffer(
                    msg.data[start_byte + z_offset:start_byte + z_offset + 4],
                    dtype=np.float32
                )[0]
                if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                    points.append([x, y, z])
        except (IndexError, ValueError):
            continue

    if not points:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float32))
    pcd = pcd.remove_non_finite_points(remove_nan=True, remove_infinite=True)
    return pcd

# ---------------------------------------------------------------------------
# Odometry / timestamp helpers
# ---------------------------------------------------------------------------

def get_odom_transform(odom_msg):
    """Extracts a 4x4 transformation matrix from a nav_msgs/Odometry message."""
    try:
        pos = odom_msg.pose.pose.position
        quat = odom_msg.pose.pose.orientation

        translation = np.array([pos.x, pos.y, pos.z], dtype=np.float64)
        rotation = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        return transform
    except (AttributeError, ValueError):
        return None


def get_closest_timestamp(timestamp, timestamps_dict):
    """Finds the closest timestamp in a dictionary to a given timestamp."""
    if not timestamps_dict:
        return None
    return min(timestamps_dict.keys(), key=lambda ts: abs(ts - timestamp))

# ---------------------------------------------------------------------------
# FPFH / registration / loop closure
# ---------------------------------------------------------------------------

def compute_fpfh_descriptor(pcd, voxel_size):
    """Computes FPFH descriptors for a point cloud."""
    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 5

    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_normal,
                max_nn=30,
            )
        )

    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_feature,
            max_nn=100,
        ),
    )
    return pcd_fpfh


def ransac_coarse_alignment(
    source, target, source_fpfh, target_fpfh, voxel_size, ransac_threshold=5
):
    """RANSAC-based coarse alignment using FPFH feature matching."""
    distance_threshold = voxel_size * ransac_threshold

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
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iteration=4000,
            confidence=0.999,
        ),
    )

    if result.fitness > 0.1:
        return result.transformation
    return None


def detect_loop_closure(
    current_frame_idx,
    current_pcd_processed,
    current_fpfh,
    accumulated_pcds,
    accumulated_fpfhs,
    accumulated_transforms,
    voxel_size,
    search_radius=10.0,
    loop_closure_fitness_thresh=0.3,
    temporal_window=100,
):
    """Detects loop closure by finding similar frames in the accumulated map."""
    loop_closures = []

    if current_frame_idx < temporal_window:
        return loop_closures

    current_pos = accumulated_transforms[current_frame_idx][:3, 3]
    searchable_indices = list(range(current_frame_idx - temporal_window))
    if not searchable_indices:
        return loop_closures

    historical_positions = np.array(
        [accumulated_transforms[i][:3, 3] for i in searchable_indices]
    )
    if len(historical_positions) == 0:
        return loop_closures

    positions_pcd = o3d.geometry.PointCloud()
    positions_pcd.points = o3d.utility.Vector3dVector(historical_positions)
    pcd_tree = o3d.geometry.KDTreeFlann(positions_pcd)

    [k, nearby_indices, _] = pcd_tree.search_radius_vector_3d(
        current_pos, search_radius
    )
    candidate_indices = [searchable_indices[i] for i in nearby_indices]
    if not candidate_indices:
        return loop_closures

    for candidate_idx in candidate_indices:
        try:
            # If it's not a keyframe, its FPFH will be None.
            candidate_fpfh = accumulated_fpfhs[candidate_idx]
            if candidate_fpfh is None:
                continue

            coarse_transform = ransac_coarse_alignment(
                current_pcd_processed,
                accumulated_pcds[candidate_idx],
                current_fpfh,
                candidate_fpfh,
                voxel_size,
            )
            if coarse_transform is None:
                continue

            reg_p2l = o3d.pipelines.registration.registration_icp(
                current_pcd_processed,
                accumulated_pcds[candidate_idx],
                voxel_size * 2,
                coarse_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20),
            )

            if reg_p2l.fitness > loop_closure_fitness_thresh:
                loop_closures.append(
                    (candidate_idx, reg_p2l.transformation, reg_p2l.fitness)
                )
        except Exception:
            continue

    return loop_closures

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_bag(args):
    """Main processing function with loop closure detection."""
    bag_path = Path(args.bag_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not bag_path.exists():
        sys.exit(f"Error: Bag file not found: {bag_path}")

    print(f"Reading bag file: {bag_path}")
    print(f"Output directory: {output_dir}")
    print(f"Point cloud topic: {args.pc_topic}")
    if args.odom_topic:
        print(f"Odometry topic: {args.odom_topic}")
    if args.imu_topic:
        print(f"IMU topic: {args.imu_topic}")
    if args.enable_loop_closure:
        print(
            f"Loop closure detection: ENABLED "
            f"(Search every {args.loop_closure_search_interval} frames)"
        )
    else:
        print("Loop closure detection: disabled")

    # --- 1. Data Extraction and Synchronization ---
    point_clouds = []
    odom_data = {}
    imu_data = {}

    topics_to_read = [args.pc_topic]
    if args.odom_topic:
        topics_to_read.append(args.odom_topic)
    if args.imu_topic:
        topics_to_read.append(args.imu_topic)

    with AnyReader([bag_path], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic in topics_to_read]
        if not connections:
            sys.exit(
                f"Error: No messages found for topics: {topics_to_read}"
            )

        print("\nExtracting messages from bag file...")
        for conn, timestamp, rawdata in tqdm(
            reader.messages(connections=connections),
            desc="Reading Messages",
        ):
            try:
                msg = reader.deserialize(rawdata, conn.msgtype)

                if conn.topic == args.pc_topic:
                    pcd = convert_ros_pc2_to_o3d(msg)
                    if pcd and len(pcd.points) > 100:
                        point_clouds.append((timestamp, pcd))

                elif conn.topic == args.odom_topic:
                    transform = get_odom_transform(msg)
                    if transform is not None:
                        odom_data[timestamp] = transform

                elif conn.topic == args.imu_topic:
                    try:
                        quat = msg.orientation
                        imu_data[timestamp] = R.from_quat(
                            [quat.x, quat.y, quat.z, quat.w]
                        ).as_matrix()
                    except (AttributeError, ValueError):
                        continue
            except Exception:
                continue

    if not point_clouds:
        sys.exit("Error: No valid point clouds were extracted.")

    print(f"\nExtracted {len(point_clouds)} point clouds")
    if args.odom_topic:
        print(f"Extracted {len(odom_data)} odometry messages")
    if args.imu_topic:
        print(f"Extracted {len(imu_data)} IMU messages")

    # --- 2. Pairwise Registration and Pose Graph Construction ---
    pose_graph = o3d.pipelines.registration.PoseGraph()
    current_transform = np.identity(4, dtype=np.float64)
    pose_graph.nodes.append(
        o3d.pipelines.registration.PoseGraphNode(current_transform)
    )

    source_pcd_raw = point_clouds[0][1]
    source_pcd_processed = source_pcd_raw.voxel_down_sample(args.voxel_size)
    source_pcd_processed.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=args.voxel_size * 2,
            max_nn=30,
        )
    )

    source_fpfh = None
    if args.enable_loop_closure:
        if 0 % args.loop_closure_search_interval == 0:
            source_fpfh = compute_fpfh_descriptor(
                source_pcd_processed, args.voxel_size
            )

    accumulated_pcds = [source_pcd_processed]
    accumulated_fpfhs = [source_fpfh]
    accumulated_transforms = [current_transform.copy()]

    previous_odom_transform = None
    if args.odom_topic:
        closest_ts = get_closest_timestamp(point_clouds[0][0], odom_data)
        if closest_ts:
            previous_odom_transform = odom_data[closest_ts]

    node_id_counter = 1
    successful_node_indices = [0]
    loop_closures_found = 0

    print("\nRegistering point clouds...")
    pbar = tqdm(range(1, len(point_clouds)), desc="Registering")
    for i in pbar:
        target_ts, target_pcd_raw = point_clouds[i]
        target_pcd_processed = target_pcd_raw.voxel_down_sample(args.voxel_size)
        target_pcd_processed.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=args.voxel_size * 2,
                max_nn=30,
            )
        )

        initial_guess = np.identity(4, dtype=np.float64)
        if args.odom_topic:
            closest_ts = get_closest_timestamp(target_ts, odom_data)
            if closest_ts:
                current_odom_transform = odom_data[closest_ts]
                if previous_odom_transform is not None:
                    initial_guess = (
                        np.linalg.inv(previous_odom_transform)
                        @ current_odom_transform
                    )
                previous_odom_transform = current_odom_transform

        try:
            reg_p2l = o3d.pipelines.registration.registration_icp(
                source_pcd_processed,
                target_pcd_processed,
                args.icp_dist_thresh,
                initial_guess,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=50
                ),
            )
        except Exception:
            continue

        if reg_p2l.fitness > args.icp_fitness_thresh:
            current_transform @= reg_p2l.transformation

            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(
                    np.linalg.inv(current_transform)
                )
            )

            information = np.eye(6) * (1.0 + reg_p2l.fitness)

            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    node_id_counter - 1,
                    node_id_counter,
                    reg_p2l.transformation,
                    information,
                    uncertain=False,
                )
            )

            accumulated_pcds.append(target_pcd_processed)
            accumulated_transforms.append(current_transform.copy())

            target_fpfh = None
            if args.enable_loop_closure and (
                i % args.loop_closure_search_interval == 0
            ):
                target_fpfh = compute_fpfh_descriptor(
                    target_pcd_processed, args.voxel_size
                )
            accumulated_fpfhs.append(target_fpfh)

            if args.enable_loop_closure and (
                i % args.loop_closure_search_interval == 0
            ):
                pbar.set_description("Registering (LC Search)")
                loop_closures = detect_loop_closure(
                    node_id_counter,
                    target_pcd_processed,
                    target_fpfh,
                    accumulated_pcds,
                    accumulated_fpfhs,
                    accumulated_transforms,
                    args.voxel_size,
                    search_radius=args.loop_closure_radius,
                    loop_closure_fitness_thresh=args.loop_closure_fitness_thresh,
                    temporal_window=100,
                )
                for (
                    lc_idx,
                    lc_transform,
                    lc_fitness,
                ) in loop_closures:
                    lc_information = np.eye(6) * lc_fitness * 100.0
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            lc_idx,
                            node_id_counter,
                            lc_transform,
                            lc_information,
                            uncertain=True,
                        )
                    )
                    loop_closures_found += 1
                pbar.set_description("Registering")

            source_pcd_processed = target_pcd_processed
            source_fpfh = target_fpfh

            node_id_counter += 1
            successful_node_indices.append(i)

    if len(pose_graph.nodes) < 2:
        sys.exit("Error: Registration failed. Too few successful registrations.")

    if args.enable_loop_closure:
        print(f"\nLoop closures detected: {loop_closures_found}")

    # --- 3. Global Optimization ---
    print("\nOptimizing the pose graph...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=args.icp_dist_thresh,
        edge_prune_threshold=0.25,
        reference_node=0,
    )

    try:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option,
        )
    except Exception as e:
        print(f"Warning: Global optimization failed: {e}")

    # --- 4. Final Map Generation and Meshing ---
    print("\nGenerating final point cloud and mesh...")

    pcd_combined = o3d.geometry.PointCloud()
    for node_idx, original_pcd_idx in tqdm(
        enumerate(successful_node_indices),
        desc="Merging Clouds",
    ):
        _, pcd_raw = point_clouds[original_pcd_idx]
        transform = pose_graph.nodes[node_idx].pose
        pcd_raw = pcd_raw.transform(transform)
        pcd_combined += pcd_raw

    if len(pcd_combined.points) == 0:
        sys.exit("Error: Combined point cloud is empty.")

    if args.level_floor:
        print("Attempting to level the floor...")
        try:
            plane_model, inliers = pcd_combined.segment_plane(
                distance_threshold=args.voxel_size * 2,
                ransac_n=3,
                num_iterations=1000,
            )
            a, b, c, d = plane_model
            normal = np.array([a, b, c])
            normal = normal / np.linalg.norm(normal)

            target_normal = np.array([0, 0, 1.0])
            if np.dot(normal, target_normal) < 0:
                normal = -normal

            v = np.cross(normal, target_normal)
            s = np.linalg.norm(v)
            if not np.isclose(s, 0):
                c = np.dot(normal, target_normal)
                vx = np.array(
                    [
                        [0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0],
                    ]
                )
                rotation_matrix = (
                    np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
                )
                pcd_combined.rotate(rotation_matrix, center=(0, 0, 0))
                print("✓ Floor leveling applied.")
            else:
                print("✓ Map is already level.")
        except Exception as e:
            print(f"Warning: Floor leveling failed: {e}")

    # Downsample before SOR
    pcd_combined_downsampled = pcd_combined.voxel_down_sample(
        voxel_size=args.voxel_size
    )

    # --- Statistical Outlier Removal BEFORE normals/meshing ---
    print("\nPerforming statistical outlier removal...")
    try:
        # You can expose these as CLI args if you want.
        nb_neighbors = 20
        std_ratio = 2.0

        pcd_clean, ind = pcd_combined_downsampled.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )
        print(
            f"Kept {len(pcd_clean.points)} points "
            f"out of {len(pcd_combined_downsampled.points)} after SOR."
        )
    except Exception as e:
        print(f"Warning: SOR failed ({e}), falling back to downsampled cloud.")
        pcd_clean = pcd_combined_downsampled

    # Save the SOR-cleaned cloud
    ply_path = output_dir / f"{bag_path.stem}_cloud.ply"
    print(f"\nSaving final point cloud to: {ply_path}")
    o3d.io.write_point_cloud(str(ply_path), pcd_clean)

    # Normal estimation for meshing uses SOR-cleaned cloud
    print("Estimating normals for meshing (on SOR-cleaned cloud)...")
    pcd_clean.estimate_normals()

    print("Reconstructing mesh using Poisson...")
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_clean,
            depth=9,
        )
       
        print(f"Initial mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")

        # --- Start: Improvements from the new code ---

        # Calculate a density threshold to remove the 5% of vertices with the lowest density.
        # This value can be exposed as a CLI argument for more control.
        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, 0.05)
        
        print(f"Removing vertices below density quantile {density_threshold:.4f}")
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        print(f"After density trimming: {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")

        # Perform a full cleanup for a more robust mesh
        print("Performing comprehensive mesh cleanup...")
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices() # Good to run this at the end.
        
        obj_path = output_dir / f"{bag_path.stem}_mesh.obj"
        print(f"Saving final mesh to: {obj_path}")
        o3d.io.write_triangle_mesh(str(obj_path), mesh)
    except Exception as e:
        print(f"Warning: Mesh generation failed: {e}")
        obj_path = None

    print("\n✓ Processing complete!")
    print(f" Point cloud: {ply_path}")
    if obj_path is not None and Path(obj_path).exists():
        print(f" Mesh: {obj_path}")
    else:
        print(" Mesh: N/A (generation failed or was skipped)")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ROS 2 Bag files with PointCloud2 data to a 3D mesh."
    )
    parser.add_argument("bag_path", help="Path to the ROS 2 bag file.")
    parser.add_argument("output_dir", help="Directory to save the output files.")
    parser.add_argument(
        "--pc_topic",
        default="/points",
        help="PointCloud2 topic name.",
    )
    parser.add_argument(
        "--odom_topic",
        default=None,
        help="Odometry topic for initial guess.",
    )
    parser.add_argument(
        "--imu_topic",
        default=None,
        help="IMU topic for orientation.",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.05,
        help="Voxel size for downsampling.",
    )
    parser.add_argument(
        "--icp_dist_thresh",
        type=float,
        default=0.2,
        help="ICP max correspondence distance.",
    )
    parser.add_argument(
        "--icp_fitness_thresh",
        type=float,
        default=0.6,
        help="Minimum ICP fitness score (0-1).",
    )
    parser.add_argument(
        "--enable_loop_closure",
        action="store_true",
        default=False,
        help="Enable loop closure detection.",
    )
    parser.add_argument(
        "--loop_closure_radius",
        type=float,
        default=10.0,
        help="Max distance to search for loop closures (m).",
    )
    parser.add_argument(
        "--loop_closure_fitness_thresh",
        type=float,
        default=0.3,
        help="Minimum ICP fitness for loop closure.",
    )
    parser.add_argument(
        "--loop_closure_search_interval",
        type=int,
        default=10,
        help="Frequency of loop closure search (every N frames).",
    )
    parser.add_argument(
        "--level_floor",
        action="store_true",
        help="Attempt to level the final map with the floor plane.",
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    process_bag(args)
