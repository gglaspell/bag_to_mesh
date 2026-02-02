#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bag_to_mesh.py with Full Loop Closure Detection (Optimized)

ROS 2 bag to 3D mesh converter with:
- Frame-to-map loop closure detection using FPFH descriptors
- RANSAC-based coarse alignment
- ICP refinement for loop closure constraints
- Improved pose graph optimization
- OPTIMIZATIONS:
    - Periodic loop closure detection (not every frame)
    - KD-Tree for fast loop closure candidate search
    - Tuned RANSAC parameters for better performance
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

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

        # Handle FLOAT32 (datatype 7)
        try:
            if x_dtype == 7 and y_dtype == 7 and z_dtype == 7:
                x = np.frombuffer(
                    msg.data[start_byte + x_offset:start_byte + x_offset + 4],
                    dtype=np.float32,
                )[0]
                y = np.frombuffer(
                    msg.data[start_byte + y_offset:start_byte + y_offset + 4],
                    dtype=np.float32,
                )[0]
                z = np.frombuffer(
                    msg.data[start_byte + z_offset:start_byte + z_offset + 4],
                    dtype=np.float32,
                )[0]

                # Only add finite points
                if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                    points.append([x, y, z])
        except (IndexError, ValueError):
            continue

    if not points:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float32))

    # Remove any remaining non-finite points
    pcd = pcd.remove_non_finite_points(remove_nan=True, remove_infinite=True)
    return pcd

# ---------------------------------------------------------------------------
# Odometry / timestamp helpers
# ---------------------------------------------------------------------------

def get_odom_transform(odom_msg):
    """
    Extracts a 4x4 transformation matrix from a nav_msgs/Odometry message.
    """
    try:
        pos = odom_msg.pose.pose.position
        quat = odom_msg.pose.pose.orientation
        translation = np.array([pos.x, pos.y, pos.z], dtype=np.float64)
        rotation = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        return transform
    except (AttributeError, ValueError) as e:
        return None


def get_closest_timestamp(timestamp, timestamps_dict):
    """
    Finds the closest timestamp in a dictionary to a given timestamp.
    """
    if not timestamps_dict:
        return None
    return min(timestamps_dict.keys(), key=lambda ts: abs(ts - timestamp))

# ---------------------------------------------------------------------------
# FPFH / registration / loop closure
# ---------------------------------------------------------------------------

def compute_fpfh_descriptor(pcd, voxel_size):
    """
    Computes FPFH (Fast Point Feature Histogram) descriptors for a point cloud.

    Args:
        pcd: Open3D PointCloud
        voxel_size: Size for radius search

    Returns:
        FPFH feature descriptor (33-dimensional)
    """
    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 5

    # Estimate normals if not present
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_normal, max_nn=30
            )
        )

    # Compute FPFH features
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_fpfh


def ransac_coarse_alignment(
    source,
    target,
    source_fpfh,
    target_fpfh,
    voxel_size,
    ransac_threshold=5,
):
    """
    RANSAC-based coarse alignment using FPFH feature matching.

    Args:
        source: Source point cloud (downsampled)
        target: Target point cloud (downsampled)
        source_fpfh: Source FPFH features
        target_fpfh: Target FPFH features
        voxel_size: Voxel size for correspondence calculation
        ransac_threshold: RANSAC correspondence threshold (multiples of voxel_size)

    Returns:
        Coarse transformation matrix (4x4) or None if alignment fails
    """
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
            # OPTIMIZATION: Reduced iterations from 50000 to 4000 for performance
            max_iteration=4000,
            confidence=0.999,
        ),
    )

    if result.fitness > 0.1:  # Minimum fitness threshold
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
    """
    Detects loop closure by finding similar frames in the accumulated map.

    Args:
        current_frame_idx: Index of current frame
        current_pcd_processed: Current point cloud (downsampled, normalized)
        current_fpfh: Current FPFH features
        accumulated_pcds: List of accumulated point clouds
        accumulated_fpfhs: List of accumulated FPFH descriptors
        accumulated_transforms: List of accumulated transforms (pose graph nodes)
        voxel_size: Voxel size
        search_radius: Max distance to search for loop closures (meters)
        loop_closure_fitness_thresh: Minimum fitness to accept loop closure
        temporal_window: Only search frames outside this many frames

    Returns:
        List of (frame_idx, transformation, fitness) tuples for detected loop closures
    """
    loop_closures = []

    if current_frame_idx < temporal_window:  # Need enough frames before searching
        return loop_closures

    # Get current robot position
    current_pos = accumulated_transforms[-1][:3, 3]

    # --- OPTIMIZATION: Use KD-Tree for fast candidate search ---
    # Create a point cloud of historical positions outside the temporal window
    searchable_indices = list(range(current_frame_idx - temporal_window))
    historical_positions = np.array([accumulated_transforms[i][:3, 3] for i in searchable_indices])
    
    if len(historical_positions) == 0:
        return loop_closures

    # Build a KD-Tree for fast neighbor search
    positions_pcd = o3d.geometry.PointCloud()
    positions_pcd.points = o3d.utility.Vector3dVector(historical_positions)
    pcd_tree = o3d.geometry.KDTreeFlann(positions_pcd)

    # Find candidate frames within the search radius
    [k, nearby_indices, _] = pcd_tree.search_radius_vector_3d(current_pos, search_radius)
    
    # Map back to original indices
    candidate_indices = [searchable_indices[i] for i in nearby_indices]

    if not candidate_indices:
        return loop_closures

    # Try RANSAC alignment with candidates
    for candidate_idx in candidate_indices:
        try:
            # Get coarse alignment using RANSAC + FPFH
            coarse_transform = ransac_coarse_alignment(
                current_pcd_processed,
                accumulated_pcds[candidate_idx],
                current_fpfh,
                accumulated_fpfhs[candidate_idx],
                voxel_size,
            )

            if coarse_transform is None:
                continue

            # Refine with ICP - use very few iterations for speed
            reg_p2l = o3d.pipelines.registration.registration_icp(
                current_pcd_processed,
                accumulated_pcds[candidate_idx],
                voxel_size * 2,
                coarse_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20),
            )

            # Accept if fitness is good
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

    # Validate input
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
        print(f"Loop closure detection: ENABLED (Search every {args.loop_closure_search_interval} frames)")
    else:
        print("Loop closure detection: disabled (use --enable_loop_closure to enable)")

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
            sys.exit(f"Error: No messages found for topics: {topics_to_read}")

        print("\nExtracting messages from bag file...")
        for conn, timestamp, rawdata in tqdm(
            reader.messages(connections=connections),
            desc="Reading Messages",
        ):
            try:
                msg = reader.deserialize(rawdata, conn.msgtype)

                if conn.topic == args.pc_topic:
                    pcd = convert_ros_pc2_to_o3d(msg)
                    if pcd and len(pcd.points) > 100:  # Filter out empty/sparse clouds
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

            except Exception as e:
                continue

    if not point_clouds:
        sys.exit("Error: No valid point clouds were extracted. Check your --pc_topic.")

    print(f"\nExtracted {len(point_clouds)} point clouds")
    if args.odom_topic:
        print(f"Extracted {len(odom_data)} odometry messages")
    if args.imu_topic:
        print(f"Extracted {len(imu_data)} IMU messages")

    # --- 2. Pairwise Registration and Pose Graph Construction with Loop Closure ---

    pose_graph = o3d.pipelines.registration.PoseGraph()
    current_transform = np.identity(4, dtype=np.float64)
    pose_graph.nodes.append(
        o3d.pipelines.registration.PoseGraphNode(current_transform)
    )

    # Pre-process the first cloud
    source_pcd_raw = point_clouds[0][1]
    source_pcd_processed = source_pcd_raw.voxel_down_sample(args.voxel_size)
    source_pcd_processed.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=args.voxel_size * 2, max_nn=30
        )
    )

    # Compute FPFH for first cloud ONLY if loop closure enabled
    source_fpfh = None
    if args.enable_loop_closure:
        source_fpfh = compute_fpfh_descriptor(source_pcd_processed, args.voxel_size)

    # Accumulate clouds and features for loop closure
    accumulated_pcds = [source_pcd_processed]
    accumulated_fpfhs = [source_fpfh]
    accumulated_transforms = [current_transform.copy()]

    previous_odom_transform = None
    if args.odom_topic:
        first_pc_ts = point_clouds[0][0]
        closest_ts = get_closest_timestamp(first_pc_ts, odom_data)
        if closest_ts:
            previous_odom_transform = odom_data[closest_ts]

    skipped_indices = set()
    loop_closures_found = 0

    print("\nRegistering point clouds...")
    pbar = tqdm(range(1, len(point_clouds)), desc="Registering")
    for i in pbar:
        target_ts, target_pcd_raw = point_clouds[i]

        # Pre-process the target cloud
        target_pcd_processed = target_pcd_raw.voxel_down_sample(args.voxel_size)
        target_pcd_processed.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=args.voxel_size * 2, max_nn=30
            )
        )

        # --- Initial Guess ---
        initial_guess = np.identity(4, dtype=np.float64)
        if args.odom_topic:
            closest_ts = get_closest_timestamp(target_ts, odom_data)
            if closest_ts:
                current_odom_transform = odom_data[closest_ts]
                if previous_odom_transform is not None:
                    # T_rel = T_prev^-1 * T_curr
                    initial_guess = (
                        np.linalg.inv(previous_odom_transform)
                        @ current_odom_transform
                    )
                previous_odom_transform = current_odom_transform

        # --- Consecutive Frame Registration (ICP) ---
        try:
            reg_p2l = o3d.pipelines.registration.registration_icp(
                source_pcd_processed,
                target_pcd_processed,
                args.icp_dist_thresh,
                initial_guess,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=50  # Fast registration
                ),
            )
        except Exception as e:
            skipped_indices.add(i)
            continue

        # --- Quality Control ---
        if reg_p2l.fitness > args.icp_fitness_thresh:
            current_transform = current_transform @ reg_p2l.transformation

            # Add node
            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(
                    np.linalg.inv(current_transform)
                )
            )

            # Create information matrix (inverse covariance) weighted by fitness
            information = np.eye(6) * (1.0 + reg_p2l.fitness)

            # Add edge between consecutive frames
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    i - 1 - len([idx for idx in skipped_indices if idx < i]),
                    i - len([idx for idx in skipped_indices if idx < i]),
                    reg_p2l.transformation,
                    information,
                    uncertain=False,
                )
            )

            # Store for loop closure detection
            accumulated_pcds.append(target_pcd_processed)
            accumulated_transforms.append(current_transform.copy())

            # Compute FPFH ONLY if loop closure enabled
            target_fpfh = None
            if args.enable_loop_closure:
                target_fpfh = compute_fpfh_descriptor(
                    target_pcd_processed, args.voxel_size
                )
            accumulated_fpfhs.append(target_fpfh)

            # --- OPTIMIZATION: Loop Closure Detection (only if enabled and at interval) ---
            if args.enable_loop_closure and (i % args.loop_closure_search_interval == 0):
                pbar.set_description("Registering (LC Search)")
                loop_closures = detect_loop_closure(
                    len(accumulated_transforms) -1,
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

                # Add loop closure edges to pose graph
                for lc_idx, lc_transform, lc_fitness in loop_closures:
                    current_node_idx = len(pose_graph.nodes) - 1
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            lc_idx,
                            current_node_idx,
                            lc_transform,
                            np.eye(6)
                            * (1.0 / (lc_fitness + 1e-9)),
                            uncertain=True, # Loop closure edges are uncertain
                        )
                    )
                    loop_closures_found += 1
                pbar.set_description("Registering")

            # Current target becomes source for next iteration
            source_pcd_processed = target_pcd_processed
            source_fpfh = target_fpfh
        else:
            skipped_indices.add(i)

    if len(pose_graph.nodes) < 2:
        sys.exit(
            "Error: Registration failed. Too few successful registrations. "
            "Try adjusting --icp_fitness_thresh or --icp_dist_thresh."
        )

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

    valid_indices = sorted(list(set(range(len(point_clouds))) - skipped_indices))

    for i in tqdm(range(len(valid_indices)), desc="Merging Clouds"):
        original_pcd_index = valid_indices[i]
        _, pcd_raw = point_clouds[original_pcd_index]
        
        node_index = i
        if node_index < len(pose_graph.nodes):
            transform = pose_graph.nodes[node_index].pose
            pcd_raw.transform(transform)
            pcd_combined += pcd_raw

    if len(pcd_combined.points) == 0:
        sys.exit("Error: Combined point cloud is empty.")

    # Optional Floor Leveling
    if args.level_floor:
        print("Attempting to level the floor...")
        try:
            # RANSAC plane segmentation to find the floor
            plane_model, inliers = pcd_combined.segment_plane(
                distance_threshold=args.voxel_size * 2,
                ransac_n=3,
                num_iterations=1000,
            )
            a, b, c, d = plane_model

            # Normal vector of the plane
            normal = np.array([a, b, c])
            normal = normal / np.linalg.norm(normal)

            # Target normal (Z-axis)
            target_normal = np.array([0, 0, 1.0])

            # If the plane is pointing down, flip it
            if np.dot(normal, target_normal) < 0:
                normal = -normal

            # Compute rotation to align the plane normal with the Z-axis
            v = np.cross(normal, target_normal)
            s = np.linalg.norm(v)
            c = np.dot(normal, target_normal)
            
            if not np.isclose(s, 0):
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                rotation_matrix = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
                pcd_combined.rotate(rotation_matrix, center=(0, 0, 0))
                print("✓ Floor leveling applied.")
            else:
                print("✓ Map is already level.")
        except Exception as e:
            print(f"Warning: Floor leveling failed: {e}")
            pass

    # Downsample the final combined cloud
    pcd_combined_downsampled = pcd_combined.voxel_down_sample(
        voxel_size=args.voxel_size
    )

    # Save final point cloud
    ply_path = output_dir / f"{bag_path.stem}_cloud.ply"
    print(f"\nSaving final point cloud to: {ply_path}")
    o3d.io.write_point_cloud(str(ply_path), pcd_combined_downsampled)

    # --- Surface Reconstruction (Meshing) ---
    print("Estimating normals for meshing...")
    pcd_combined_downsampled.estimate_normals()

    print("Reconstructing mesh using Poisson...")
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd_combined_downsampled, depth=9
        )

        # Clean up the mesh - remove low density vertices
        print("Cleaning mesh...")
        densities = np.asarray(densities)
        density_threshold = np.quantile(densities, 0.05)
        vertices_to_remove = densities < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)

        # Remove degenerate triangles and unreferenced vertices
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()

        # Save final mesh
        obj_path = output_dir / f"{bag_path.stem}_mesh.obj"
        print(f"Saving final mesh to: {obj_path}")
        o3d.io.write_triangle_mesh(str(obj_path), mesh)
    except Exception as e:
        print(f"Warning: Mesh generation failed: {e}")
        print("Continuing with point cloud output only.")

    print("\n✓ Processing complete!")
    print(f"  Point cloud: {ply_path}")
    if "obj_path" in locals() and Path(obj_path).exists():
        print(f"  Mesh: {obj_path}")
    else:
        print("  Mesh: N/A (generation failed or was skipped)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Convert ROS 2 Bag files with PointCloud2 data to PLY and OBJ files. "
            "Includes optional loop closure detection for accurate SLAM mapping."
        )
    )

    parser.add_argument("bag_path", help="Path to the ROS 2 bag file.")
    parser.add_argument("output_dir", help="Directory to save the output files.")

    # Topic Arguments
    parser.add_argument(
        "--pc_topic",
        default="/points",
        help="PointCloud2 topic name. (default: /points)",
    )
    parser.add_argument(
        "--odom_topic",
        default=None,
        help="Odometry topic name for initial guess. (optional)",
    )
    parser.add_argument(
        "--imu_topic",
        default=None,
        help="IMU topic name for orientation. (optional)",
    )

    # Algorithm Tuning Arguments
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.05,
        help="Voxel size for downsampling. Smaller = more detail, slower. (default: 0.05)",
    )
    parser.add_argument(
        "--icp_dist_thresh",
        type=float,
        default=0.2,
        help="ICP max correspondence distance. (default: 0.2)",
    )
    parser.add_argument(
        "--icp_fitness_thresh",
        type=float,
        default=0.6,
        help=(
            "Minimum ICP fitness score (0-1) to accept a transformation. "
            "Higher = stricter. (default: 0.6)"
        ),
    )

    # Loop Closure Arguments
    parser.add_argument(
        "--enable_loop_closure",
        action="store_true",
        default=False,
        help="Enable loop closure detection. (default: disabled for speed)",
    )
    parser.add_argument(
        "--loop_closure_radius",
        type=float,
        default=10.0,
        help="Maximum distance to search for loop closures in meters. (default: 10.0)",
    )
    parser.add_argument(
        "--loop_closure_fitness_thresh",
        type=float,
        default=0.3,
        help=(
            "Minimum ICP fitness for loop closure acceptance (0-1). "
            "Higher = stricter. (default: 0.3)"
        ),
    )
    # --- NEW ARGUMENT ---
    parser.add_argument(
        "--loop_closure_search_interval",
        type=int,
        default=10,
        help="Frequency of loop closure search (every N frames). (default: 10)",
    )

    parser.add_argument(
        "--level_floor",
        action="store_true",
        help="Attempt to level the final map with the floor plane.",
    )

    # Display help if no arguments
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # Validate argument ranges
    if not (0 < args.voxel_size < 1.0):
        sys.exit("Error: --voxel_size must be between 0 and 1.0")
    if not (0 < args.icp_dist_thresh < 10.0):
        sys.exit("Error: --icp_dist_thresh must be between 0 and 10.0")
    if not (0.0 <= args.icp_fitness_thresh <= 1.0):
        sys.exit("Error: --icp_fitness_thresh must be between 0.0 and 1.0")
    if not (0.0 <= args.loop_closure_fitness_thresh <= 1.0):
        sys.exit("Error: --loop_closure_fitness_thresh must be between 0.0 and 1.0")
    if args.loop_closure_search_interval < 1:
        sys.exit("Error: --loop_closure_search_interval must be at least 1.")

    process_bag(args)

