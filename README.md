# ROS 2 Bag to 3D Mesh Converter

Convert ROS 2 bag files containing LiDAR point clouds into high-quality 3D models with precision-tuned registration and drift correction.

---

## 📋 What This Does

This tool transforms ROS 2 bag files with `sensor_msgs/PointCloud2` data into:
- **Point Cloud** (`.ply`) - Dense 3D representation of your environment
- **Surface Mesh** (`.obj`) - Textured 3D model ready for visualization or simulation

**Design Philosophy:** Accuracy over speed. Built for offline post-processing where quality matters more than real-time performance.

### Key Capabilities
- **Point-to-Plane ICP Registration** - Sub-centimeter alignment accuracy
- **Global Pose Graph Optimization** - Corrects accumulated drift across entire trajectories
- **Odometry Integration** - Uses wheel/visual odometry for better initial alignment
- **Loop Closure Detection** (Optional) - Finds and constrains revisited areas with FPFH feature matching
- **Z-Axis Leveling** - Prevents vertical drift in planar environments
- **Adaptive Frame Filtering** - Automatically skips low-quality registrations

---

## 🛠 What You'll Need

### Required
- **Docker** ([Install Docker](https://docs.docker.com/get-docker/))
- **ROS 2 Bag File** with `sensor_msgs/PointCloud2` messages
- **8GB+ RAM** (16GB+ recommended for large datasets)

### Optional (Improves Results)
- Odometry topic (`nav_msgs/Odometry`) - For better initial alignment
- ~5-10GB free disk space per hour of bag data

---

## 🚀 Getting Started

### 1. Set Up Your Project

Create a project directory and add the necessary files:

```bash
mkdir bag-to-mesh && cd bag-to-mesh
# Place Dockerfile and bag_to_mesh.py here
```

### 2. Build the Docker Image

```bash
docker build -t bag-to-mesh .
```

This downloads dependencies (~2GB) and takes 5-10 minutes on first build.

### 3. Organize Your Data

Create a data directory structure:

```bash
mkdir -p data/input data/output
# Copy your ROS 2 bag file to data/input/
```

### 4. Run Your First Conversion

**Basic command** (assumes point cloud topic is `/points`):

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  bag-to-mesh \
  /app/data/input/your_bag_file \
  /app/data/output
```

**Recommended command** (with odometry and optimizations):

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  bag-to-mesh \
  /app/data/input/your_bag_file \
  /app/data/output \
  --pc_topic /your/pointcloud/topic \
  --odom_topic /your/odom/topic \
  --icp_fitness_thresh 0.4 \
  --level_floor
```

### 5. View Your Results

Output files will be in `data/output/`:
- `your_bag_file_cloud.ply` - Point cloud (view in CloudCompare, MeshLab)
- `your_bag_file_mesh.obj` - 3D mesh (view in Blender, Unity, etc.)

---

## 💡 Common Use Cases

### Indoor Mapping with TurtleBot

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  bag-to-mesh \
  /app/data/tb3_office_scan \
  /app/data/output \
  --pc_topic /scan/points \
  --odom_topic /odom \
  --voxel_size 0.02 \
  --icp_fitness_thresh 0.5 \
  --level_floor
```

**Why these settings?**
- `voxel_size 0.02` - Finer detail for small spaces
- `icp_fitness_thresh 0.5` - Balanced quality (indoor has good features)
- `level_floor` - Flattens floors in multi-room scans

---

### Outdoor Mapping with DLIO/LIO-SAM

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  bag-to-mesh \
  /app/data/outdoor_survey \
  /app/data/output \
  --pc_topic /dlio/odom_node/pointcloud/deskewed \
  --odom_topic /dlio/odom_node/odom \
  --icp_fitness_thresh 0.2 \
  --icp_dist_thresh 0.5 \
  --level_floor \
  --voxel_size 0.1
```

**Why these settings?**
- Lower fitness threshold (0.2) - Outdoor has fewer distinct features
- Higher distance threshold (0.5) - Accommodates larger environments
- Uses deskewed clouds from SLAM system for better initial quality

---

### High-Quality Object Scanning

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  bag-to-mesh \
  /app/data/object_scan \
  /app/data/output \
  --pc_topic /camera/depth/points \
  --voxel_size 0.005 \
  --icp_dist_thresh 0.05 \
  --icp_fitness_thresh 0.7
```

**Why these settings?**
- `voxel_size 0.005` - Maximum detail preservation
- Tight thresholds - Ensures precise alignment for small objects

---

### Large Campus/Warehouse Mapping (with Loop Closure)

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  bag-to-mesh \
  /app/data/warehouse_full \
  /app/data/output \
  --pc_topic /velodyne_points \
  --odom_topic /integrated_odom \
  --voxel_size 0.1 \
  --icp_fitness_thresh 0.3 \
  --level_floor \
  --enable_loop_closure
```

**Why these settings?**
- `voxel_size 0.1` - Aggressive downsampling for large environments
- `enable_loop_closure` - Constrains drift when revisiting areas
- `icp_fitness_thresh 0.3` - More permissive to include more frames

---

## 📖 Parameter Quick Reference

| Parameter                      | Default  | Range          | Purpose                                                                                                                                                        |
| ------------------------------ | -------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| bag_path                       | required | -              | Path to ROS 2 bag file                                                                                                                                         |
| output_dir                     | required | -              | Where to save output files                                                                                                                                     |
| --pc_topic                     | /points  | -              | PointCloud2 topic name                                                                                                                                         |
| --odom_topic                   | None     | -              | Odometry topic (nav_msgs/Odometry)                                                                                                                             |
| --voxel_size                   | 0.05     | 0.001-1.0      | Downsampling resolution (meters)                                                                                                                               |
| --icp_dist_thresh              | 0.2      | 0.01-10.0      | Max point correspondence distance (m)                                                                                                                          |
| --icp_fitness_thresh           | 0.6      | 0.0-1.0        | Min % of points aligned to accept frame                                                                                                                        |
| --enable_loop_closure          | False    | -              | Enable loop closure detection                                                                                                                                  |
| --loop_closure_radius          | 10.0     | 1.0-50.0       | Search radius for loop closure (meters)                                                                                                                        |
| --loop_closure_fitness_thresh  | 0.3      | 0.0-1.0        | Min fitness for loop closure acceptance                                                                                                                        |
| --loop_closure_search_interval | 10       | -              | Frequency of loop closure search (every N frames)                                                                                                              |
| --level_floor                  | False    | -              | Apply post-processing Z-leveling                                                                                                                               |
| --decimate_target              | None     | 0.01-1.0 or >1 | Reduce mesh triangles after reconstruction. Values ≤ 1.0 = ratio to keep (e.g. 0.25 = keep 25%); values > 1 = absolute triangle count. Omit to skip decimation |
| --odom_max_latency        | 0.5 s   | Staleness cutoff for odom↔pointcloud timestamp matching |
| --poisson_depth           | 9       | Octree depth for Poisson reconstruction                 |
| --density_trim_percentile | 0.05    | Bottom fraction of low-density vertices to remove       |

---

## 🎯 Pro Tips

### Finding Your Topics

Don't know your topic names? Run this on your bag file:

```bash
ros2 bag info /path/to/your/bag_file
```

Look for:
- `sensor_msgs/msg/PointCloud2` - Your point cloud topic
- `nav_msgs/msg/Odometry` - Your odometry topic

### Performance Tuning

**Speed vs Quality Trade-offs:**

| Goal | Voxel Size | Fitness Thresh | ICP Distance |
|------|------------|----------------|--------------| 
| 🚀 Fast Preview | 0.1 | 0.3 | 0.5 |
| ⚖️ Balanced | 0.05 | 0.5 | 0.2 |
| 🎨 Maximum Quality | 0.01 | 0.7 | 0.1 |

### Loop Closure (Disabled by Default)

**Important:** Loop closure detection is now **disabled by default** for speed. Most use cases don't need it.

Enable it only for **large-scale mapping** where you revisit areas:

```bash
--enable_loop_closure
```

**When to use:**
- ✅ Large outdoor surveys (>30 min of data)
- ✅ Multi-floor indoor mapping
- ✅ Areas with significant loops/revisits
- ✅ When accuracy is critical

**When to skip:**
- ❌ Linear paths (corridors, roads)
- ❌ Small rooms/areas
- ❌ When speed is critical

**Performance impact:**
- Disabling (default): ~2-3 min per 1000 frames
- Enabling: ~5-15 min per 1000 frames (3-8x slower)

### When to Use --level_floor

✅ **Use it when:**
- Mapping single-floor indoor spaces
- Operating on parking lots, warehouses
- You see undulating floors in the output

❌ **Don't use when:**
- Mapping multi-story buildings
- Scanning terrain/hills
- Working with ramps or significant elevation changes

---

## 🔧 Troubleshooting

### "Error: No messages found for topics"

**Problem:** Topic names don't match your bag file.

**Fix:**
```bash
# Check available topics
ros2 bag info your_bag_file

# Use exact topic names from output
docker run --rm -v "$(pwd)/data:/app/data" bag-to-mesh \
  /app/data/your_bag \
  /app/data/output \
  --pc_topic /exact/topic/name
```

---

### "Error: Registration failed. Try..."

**Problem:** Too few successful registrations (frames being rejected).

**Fix (in order of priority):**

1. **Lower fitness threshold** - Accept more frames
   ```bash
   --icp_fitness_thresh 0.3
   ```

2. **Increase distance threshold** - Allow looser matching
   ```bash
   --icp_dist_thresh 0.5
   ```

3. **Check point cloud quality** - Ensure your LiDAR data is valid:
   ```bash
   ros2 topic echo /your/pc/topic --once
   ```

---

### Point Clouds Look "Stacked" or Doubled

**Problem:** Severe registration failure causing ghost geometry.

**Fix:**
```bash
# Much stricter filtering
--icp_fitness_thresh 0.7 \
--icp_dist_thresh 0.15
```

This rejects more frames but ensures remaining ones are clean.

---

### Processing is Too Slow

**Problem:** Registration taking hours on large datasets.

**Speed-up strategies (in priority order):**

1. **Increase voxel size** (biggest impact):
   ```bash
   --voxel_size 0.1
   ```

2. **Reduce ICP iterations** - Already at 50 by default (fast)

3. **Disable loop closure** (if enabled):
   ```bash
   # Don't use --enable_loop_closure
   ```

4. **Filter bag file first** (pre-process):
   ```bash
   ros2 bag filter input.bag output.bag \
     "topic == '/your/topic' and t.sec % 5 == 0"
   ```
   This keeps every 5th second of data.

5. **Check odometry sync** - Verify `--odom_topic` is correct (speeds convergence)

---

### Vertical Drift (Z-axis Issues)

**Problem:** Floors slope or multi-level structure compressed/stretched.

**Fixes (in order):**

1. **Try Z-leveling** (for flat environments):
   ```bash
   --level_floor
   ```

2. **Use loop closure** (constraints help):
   ```bash
   --enable_loop_closure
   ```

3. **Reduce fitness threshold** (include more frames):
   ```bash
   --icp_fitness_thresh 0.3
   ```

---

### "Mesh generation failed"

**Problem:** Point cloud too sparse or irregular for Poisson reconstruction.

**Workaround:**
- Use the `.ply` point cloud file instead
- Try external meshing tools (MeshLab's "Surface Reconstruction: Poisson")
- Increase point density: `--voxel_size 0.02`

---

## ⚠️ Known Limitations

### Loop Closure
- Searches within spatial radius and temporal window
- Not designed for GPS-denied environments with severe ambiguity
- May produce false positives if environments are highly repetitive
- **Currently disabled by default due to performance impact**

### Scale
- Tested up to **5000 frames** (~10 min of data at 10 Hz)
- Larger datasets may require **32GB+ RAM**
- Consider splitting very large bags

### Point Cloud Types
- **Only works with XYZ fields** (FLOAT32)
- Intensity/color fields ignored
- Cannot efficiently process organized point clouds with NaN values

### Real-Time
- Not designed for real-time operation
- Typical processing speed: **~2-3 min per minute of recorded data** (without loop closure)

### Environment Constraints
- Best for **feature-rich environments** (indoor spaces, structured outdoor)
- Struggles with:
  - Long hallways without distinct features
  - Completely repetitive geometry
  - Low-density point clouds (<1000 points/frame)

---

## 📊 Performance Guide

### Processing Times (Without Loop Closure)

| Dataset Size | Point Clouds | Voxel Size | Time | Peak RAM |
|-------------|--------------|------------|------|----------|
| Small room | ~500 frames | 0.02m | 1-2 min | 4 GB |
| Office floor | ~1500 frames | 0.05m | 3-5 min | 8 GB |
| Large warehouse | ~3000 frames | 0.1m | 8-12 min | 12 GB |
| Campus outdoor | ~5000 frames | 0.1m | 15-25 min | 16 GB |

### Processing Times (With Loop Closure Enabled)

| Dataset Size | Point Clouds | Voxel Size | Time | Peak RAM |
|-------------|--------------|------------|------|----------|
| Office floor | ~1500 frames | 0.05m | 10-20 min | 8 GB |
| Large warehouse | ~3000 frames | 0.1m | 30-60 min | 12 GB |
| Campus outdoor | ~5000 frames | 0.1m | 60-120 min | 16 GB |

**Notes:**
- Times depend heavily on CPU core count
- Loop closure adds 3-8x processing time
- Smaller voxel sizes increase processing time exponentially

### Hardware Recommendations

| Scenario | CPU | RAM | Storage |
|----------|-----|-----|---------|
| Fast preview | 4 cores | 8 GB | 5 GB |
| Standard mapping | 8 cores | 16 GB | 20 GB |
| Large datasets | 16+ cores | 32 GB | 50+ GB SSD |

### Optimization Priority

**If you need faster processing:**
1. Increase `--voxel_size` to 0.1 (biggest speedup)
2. Don't use `--enable_loop_closure` (default: disabled)
3. Lower `--icp_fitness_thresh` to 0.4 (skips fewer frames)
4. Pre-filter your bag file (reduce frame count)

**If you need better quality:**
1. Decrease `--voxel_size` to 0.02
2. Add odometry: `--odom_topic`
3. Increase `--icp_fitness_thresh` to 0.7
4. Enable `--level_floor` for indoor
5. Use `--enable_loop_closure` for large areas with loops

---

## 📝 Output Files Explained

### `*_cloud.ply`
- **Type:** Point cloud
- **Use:** Visualization, measurements, further processing
- **Tools:** CloudCompare, MeshLab, Open3D
- **Size:** ~500MB per 1000 frames (voxel 0.05m)

### `*_mesh.obj`
- **Type:** Triangle mesh (Poisson surface reconstruction)
- **Use:** Rendering, game engines, simulations
- **Tools:** Blender, Unity, Unreal Engine
- **Size:** ~100MB per 1000 frames

---

## 🤝 Getting Help

**Check your logs for these key metrics:**

```
Extracted X point clouds          ← Should be >100
Registering: Y%                   ← Progress indicator
Loop closures detected: Z         ← Shows if enabled (default: 0)
```

If registration is slow or failing, try:
1. Lower `--icp_fitness_thresh` to 0.3
2. Increase `--icp_dist_thresh` to 0.5
3. Add `--odom_topic` if available

---

## 📦 Project Files

- `bag_to_mesh.py` - Main conversion script (optimized for speed)
- `Dockerfile` - Container build instructions
- `README.md` - This file

---

## 🔗 Viewing Your Results

### Point Clouds (`.ply`)
- **CloudCompare** (free, cross-platform): https://www.cloudcompare.org/
- **MeshLab** (free): https://www.meshlab.net/

### Meshes (`.obj`)
- **Blender** (free, powerful): https://www.blender.org/
- **Online viewer**: https://3dviewer.net/

---

**Built with:** Open3D • NumPy • SciPy • rosbags • Python 3.10
