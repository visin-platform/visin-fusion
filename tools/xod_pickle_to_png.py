#!/usr/bin/env python3
"""
Script to convert XOD LiDAR pickle files to L2D (LiDAR-to-2D) projections.
Creates PNG visualizations of LiDAR point clouds projected onto camera images.

Based on Waymo approach but adapted for XOD dataset:
- Uses pre-computed camera coordinates from pickle files
- Scales coordinates from XOD's coordinate system to output resolution
- Fixed output size to match camera images and annotations

This script handles:
1. Loading LiDAR data from pickle files
2. Converting XOD camera coordinates to proper scale
3. Creating RGB visualizations of X, Y, Z channels
4. Saving as PNG files for dataset preparation

Camera Intrinsics (XOD - scaled for 1363x768 output):
- Based on Waymo parameters, adjusted for XOD resized camera size (1363x768)
- fx: 756.0 (focal length x, scaled)
- fy: 756.0 (focal length y, scaled)
- cx: 681.5 (principal point x, scaled)
- cy: 324.0 (principal point y, scaled)

Output Size: Fixed at 1363x768 to match resized camera dimensions

LiDAR Normalization (from existing code):
- mean: [-0.17263354, 0.85321806, 24.5527253]
- std: [7.34546552, 1.17227659, 15.83745082]
"""

import os
import pickle
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


class XODL2DProjector:
    """Handles LiDAR to 2D camera projection for XOD dataset."""

    def __init__(self, output_width=1363, output_height=768):
        """
        Initialize the projector.

        Args:
            output_width: Width of output projection image (fixed at 1363 to match resized camera size)
            output_height: Height of output projection image (fixed at 768 to match resized camera size)
        """
        self.output_width = output_width
        self.output_height = output_height

        # XOD camera intrinsics (scaled for 1363x768 output)
        # Based on Waymo parameters, scaled to match resized camera dimensions
        # Original Waymo camera is ~1920x1280, intrinsics scaled for 1363x768
        # fx/fy = 1260 * (768/1280) ≈ 1260 * 0.6 = 756
        # cx/cy = 960 * (1363/1920) ≈ 960 * 0.71 = 682, 540 * (768/1280) ≈ 540 * 0.6 = 324
        self.camera_intrinsics = {
            "fx": 756.0,  # focal length x (scaled to match resized camera size)
            "fy": 756.0,  # focal length y (scaled to match resized camera size)
            "cx": 681.5,  # principal point x (scaled to match resized camera size)
            "cy": 324.0,  # principal point y (scaled to match resized camera size)
        }

        # LiDAR normalization parameters (same as Waymo)
        self.lidar_mean = np.array([-0.17263354, 0.85321806, 24.5527253])
        self.lidar_std = np.array([7.34546552, 1.17227659, 15.83745082])

    def load_pickle_data(self, pickle_path):
        """
        Load LiDAR data from XOD pickle file.

        Args:
            pickle_path: Path to pickle file

        Returns:
            points3d: 3D LiDAR points (N, 3)
            camera_coord_scaled: 2D camera coordinates (N, 2) - scaled to output space
            camera_coord_original: 2D camera coordinates (N, 2) - in original camera space
        """
        with open(pickle_path, "rb") as f:
            lidar_data = pickle.load(f)

        points3d = lidar_data["3d_points"]

        # Check if camera coordinates are available
        if "camera_coordinates" in lidar_data:
            camera_coord = lidar_data["camera_coordinates"]
            # XOD camera coordinates are stored as [camera_id, u, v]
            # Filter for front camera (camera 1, similar to Waymo)
            mask = camera_coord[:, 0] == 1
            points3d = points3d[mask]
            camera_coord_original = camera_coord[mask, 1:3]  # u, v coordinates in camera space

            # XOD camera coordinates are in camera image space (4240x2824)
            # Scale to output space (1363x768)
            u_coords = camera_coord_original[:, 0].astype(np.float32)
            v_coords = camera_coord_original[:, 1].astype(np.float32)

            # Scale from camera space (4240x2824) to output space (1363x768)
            u_output = u_coords * (self.output_width / 4240.0)
            v_output = v_coords * (self.output_height / 2824.0)

            camera_coord_scaled = np.stack([u_output, v_output], axis=1)
            return points3d, camera_coord_scaled, camera_coord_original
        raise ValueError(f"No camera coordinates found in {pickle_path}")

    def normalize_lidar_points(self, points3d):
        """
        Normalize LiDAR points using dataset statistics.

        Args:
            points3d: 3D points (N, 3) with columns [Z, X, Y]

        Returns:
            normalized_points: Normalized points (N, 3) as [X, Y, Z]
        """
        # Reorder to [X, Y, Z] and normalize
        x_lid = (points3d[:, 1] - self.lidar_mean[0]) / self.lidar_std[0]
        y_lid = (points3d[:, 2] - self.lidar_mean[1]) / self.lidar_std[1]
        z_lid = (points3d[:, 0] - self.lidar_mean[2]) / self.lidar_std[2]

        return np.stack([x_lid, y_lid, z_lid], axis=1)

    def create_l2d_projection(self, points3d, camera_coord):
        """
        Create L2D projection image from LiDAR points.

        Args:
            points3d: 3D LiDAR points (N, 3)
            camera_coord: Pre-computed camera coordinates (N, 2) in output space

        Returns:
            X_img, Y_img, Z_img: PIL Images for each channel
        """
        # Normalize points
        normalized_points = self.normalize_lidar_points(points3d)

        # Create projection images
        X = np.zeros((self.output_height, self.output_width), dtype=np.float32)
        Y = np.zeros((self.output_height, self.output_width), dtype=np.float32)
        Z = np.zeros((self.output_height, self.output_width), dtype=np.float32)

        # Get valid coordinates within image bounds
        rows = np.floor(camera_coord[:, 1]).astype(int)
        cols = np.floor(camera_coord[:, 0]).astype(int)

        # Filter valid coordinates
        valid_mask = (rows >= 0) & (rows < self.output_height) & (cols >= 0) & (cols < self.output_width)

        rows = rows[valid_mask]
        cols = cols[valid_mask]
        points = normalized_points[valid_mask]

        # Project points to image
        X[rows, cols] = points[:, 0]
        Y[rows, cols] = points[:, 1]
        Z[rows, cols] = points[:, 2]

        # Convert to PIL Images (convert to 8-bit grayscale for PNG)
        # Use robust normalization to prevent single-channel dominance
        # Use percentile-based normalization to handle outliers

        def robust_normalize(channel, percentile=95):
            """Normalize channel using percentiles to handle outliers."""
            abs_max = np.percentile(np.abs(channel[channel != 0]), percentile)
            if abs_max > 0:
                # Normalize to -1 to +1 range, then scale to 0-255
                normalized = np.clip(channel / abs_max, -1, 1)
                return ((normalized + 1) / 2 * 255).astype(np.uint8)
            return np.zeros_like(channel, dtype=np.uint8)

        X_norm = robust_normalize(X)
        Y_norm = robust_normalize(Y)
        Z_norm = robust_normalize(Z)

        X_img = Image.fromarray(X_norm, mode="L")
        Y_img = Image.fromarray(Y_norm, mode="L")
        Z_img = Image.fromarray(Z_norm, mode="L")

        return X_img, Y_img, Z_img

    def create_overlay_visualization(self, camera_path, lidar_png_path, output_path, points3d, camera_coord, alpha=0.9):
        """
        Create overlay visualization of LiDAR projection on camera image.

        Args:
            camera_path: Path to camera image
            lidar_png_path: Path to LiDAR PNG projection
            output_path: Path to save overlaid image
            points3d: 3D LiDAR points (N, 3)
            camera_coord: Pre-computed camera coordinates (N, 2) in camera space
            alpha: Transparency alpha for LiDAR overlay (0-1)
        """
        # Load camera image
        camera_img = Image.open(camera_path).convert("RGBA")
        camera_width, camera_height = camera_img.size

        # Calculate distances for each point
        distances = np.linalg.norm(points3d, axis=1)

        # Create distance-based image at camera resolution

        # Use camera coordinates directly (they are already in camera space)
        rows = np.round(camera_coord[:, 1]).astype(int)  # v coordinates
        cols = np.round(camera_coord[:, 0]).astype(int)  # u coordinates

        # Filter valid coordinates
        valid_mask = (rows >= 0) & (rows < camera_height) & (cols >= 0) & (cols < camera_width)
        rows = rows[valid_mask]
        cols = cols[valid_mask]
        valid_distances = distances[valid_mask]

        # Project distances to image with enhanced contrast
        if valid_distances.size > 0:
            # Use log scaling for better contrast, but enhance the color mapping
            log_distances = np.log(valid_distances + 1.0)

            # Normalize with enhanced contrast - use percentiles for better color distribution
            dist_min = np.percentile(log_distances, 5)  # Ignore outliers
            dist_max = np.percentile(log_distances, 95)  # Ignore outliers

            if dist_max > dist_min:
                # Enhanced normalization with gamma correction for better contrast
                normalized = np.clip((log_distances - dist_min) / (dist_max - dist_min), 0, 1)
                # Apply gamma correction to enhance color differences
                gamma = 0.7  # < 1 makes dark areas lighter, enhancing contrast
                normalized_gamma = np.power(normalized, gamma)
                # Ensure no NaN values
                normalized_gamma = np.nan_to_num(normalized_gamma, nan=0.5)
                distance_norm = (normalized_gamma * 255).astype(np.uint8)
            else:
                distance_norm = np.full_like(log_distances, 127, dtype=np.uint8)

            dist_img = np.zeros((camera_height, camera_width), dtype=np.uint8)
            dist_img[rows, cols] = distance_norm
        else:
            dist_img = np.zeros((camera_height, camera_width), dtype=np.uint8)

        # Apply JET colormap (better for distance visualization - starts with blue, goes to red)
        lidar_colored = cv2.applyColorMap(dist_img, cv2.COLORMAP_JET)
        lidar_colored_rgb = cv2.cvtColor(lidar_colored, cv2.COLOR_BGR2RGB)

        # Convert camera to numpy array for drawing (use RGB, not RGBA)
        camera_array = np.array(camera_img.convert("RGB"))

        # Get coordinates of LiDAR points
        point_coords = np.where(dist_img > 0)
        rows, cols = point_coords

        # Draw larger circles for each LiDAR point
        for r, c in zip(rows, cols):
            # Get the color for this point from the colormap
            color = lidar_colored_rgb[r, c]
            # Draw a circle with radius 5 pixels (more visible)
            cv2.circle(camera_array, (c, r), 5, color.tolist(), -1)  # -1 fills the circle

        # Convert back to PIL and save
        overlay_img = Image.fromarray(camera_array)
        overlay_img.save(output_path)

    def create_combined_lidar_png(self, X_img, Y_img, Z_img, save_path):
        """
        Create a combined 3-channel PNG file for training.

        Args:
            X_img, Y_img, Z_img: Individual channel PIL Images
            save_path: Path to save the combined PNG
        """
        # Convert to numpy arrays
        X = np.array(X_img).astype(np.float32) / 255.0
        Y = np.array(Y_img).astype(np.float32) / 255.0
        Z = np.array(Z_img).astype(np.float32) / 255.0

        # Stack into 3-channel image
        combined_array = np.stack([X, Y, Z], axis=2)

        # Convert to uint8
        combined_uint8 = (combined_array * 255).astype(np.uint8)

        # Create PIL Image
        combined_img = Image.fromarray(combined_uint8, mode="RGB")
        combined_img.save(save_path)

    def process_pickle_file(self, pickle_path, output_dir, create_visualization=False, camera_path=None):
        """
        Process a single pickle file and save projections.

        Args:
            pickle_path: Path to input pickle file
            output_dir: Directory to save output images
            create_visualization: Whether to create RGB visualization
            camera_path: Path to corresponding camera image (required for visualization)
        """
        # Load data
        points3d, camera_coord_scaled, camera_coord_original = self.load_pickle_data(pickle_path)

        if len(points3d) == 0:
            print(f"Warning: No points found in {pickle_path}")
            return

        # Create projections
        X_img, Y_img, Z_img = self.create_l2d_projection(points3d, camera_coord_scaled)

        # Get output filename
        pickle_name = Path(pickle_path).stem
        output_base = os.path.join(output_dir, pickle_name)

        # Create combined 3-channel PNG
        self.create_combined_lidar_png(X_img, Y_img, Z_img, f"{output_base}.png")

        # Create overlay visualization if requested
        if create_visualization and camera_path and os.path.exists(camera_path):
            visualize_output_path = f"{output_base}_overlay.png"
            self.create_overlay_visualization(
                camera_path, f"{output_base}.png", visualize_output_path, points3d, camera_coord_original
            )

    def process_directory(self, input_dir, create_visualization=False):
        """
        Process all pickle files in a directory.

        Args:
            input_dir: Directory containing pkl/ subdirectory with pickle files
            create_visualization: Whether to create overlay visualizations
        """
        input_path = Path(input_dir)

        # Check if pkl subdirectory exists
        pkl_dir = input_path / "pkl"
        if not pkl_dir.exists():
            print(f"Error: {pkl_dir} does not exist")
            return

        # Create output directory
        output_dir = input_path / "lidar_png"
        output_dir.mkdir(exist_ok=True)

        # Find all pickle files
        pickle_files = list(pkl_dir.glob("*.pkl"))
        if not pickle_files:
            print(f"No pickle files found in {pkl_dir}")
            return

        print(f"Found {len(pickle_files)} pickle files in {pkl_dir}")
        print(f"Output directory: {output_dir}")

        # Process each pickle file
        processed_count = 0
        for pickle_file in tqdm(pickle_files, desc="Processing pickle files"):
            try:
                # Find corresponding camera image for visualization
                camera_path = None
                if create_visualization:
                    camera_file = input_path / "rgb" / f"{pickle_file.stem}.png"
                    if camera_file.exists():
                        camera_path = str(camera_file)

                # Process the file
                self.process_pickle_file(str(pickle_file), str(output_dir), create_visualization, camera_path)
                processed_count += 1

            except Exception as e:
                print(f"Error processing {pickle_file}: {e}")
                continue

        print(f"Processing complete. Processed {processed_count}/{len(pickle_files)} files.")

    def process_from_file_list(self, file_list_path, dataset_root="", output_root="", create_visualization=False):
        # Read file list
        with open(file_list_path) as f:
            camera_paths = [line.strip() for line in f if line.strip()]

        if not camera_paths:
            print(f"No paths found in {file_list_path}")
            return

        print(f"Found {len(camera_paths)} camera paths to process")

        # Infer dataset root
        if not dataset_root:
            file_list_path_obj = Path(file_list_path)
            if "all.txt" in str(file_list_path_obj):
                dataset_root = str(file_list_path_obj.parent)

        print(f"Using dataset root: {dataset_root}")
        if output_root is None:
            output_root = dataset_root
        print(f"Using output root: {output_root}")

        # Convert camera paths to pickle paths and determine output paths
        pickle_paths = []
        output_dirs = []

        for cam_path in camera_paths:
            # Prepend dataset root
            full_cam_path = os.path.join(dataset_root, cam_path)

            # Replace /camera/ with /pkl/ and .png with .pkl
            lidar_path = full_cam_path.replace("/camera/", "/pkl/").replace(".png", ".pkl")
            pickle_paths.append(lidar_path)

            # Create output path: replace /camera/ with /lidar_png/
            output_path = full_cam_path.replace("/camera/", "/lidar_png/").replace(".png", ".png")
            if output_root and output_root != dataset_root:
                output_path = output_path.replace(dataset_root, output_root)
            elif not output_root:
                # Remove dataset_root prefix when output_root is empty
                output_path = output_path.replace(dataset_root + "/", "")
            output_dir = os.path.dirname(output_path)
            output_dirs.append(output_dir)

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Create visualization directory if needed
            if create_visualization:
                visualize_output_path = full_cam_path.replace("/camera/", "/lidar_png_visualize/").replace(
                    ".png", "_overlay.png"
                )
                if output_root and output_root != dataset_root:
                    visualize_output_path = visualize_output_path.replace(dataset_root, output_root)
                elif not output_root:
                    # Remove dataset_root prefix when output_root is empty
                    visualize_output_path = visualize_output_path.replace(dataset_root + "/", "")
                visualize_dir = os.path.dirname(visualize_output_path)
                os.makedirs(visualize_dir, exist_ok=True)

        # Process each pickle file
        processed_count = 0
        for i, pickle_path in enumerate(tqdm(pickle_paths, desc="Processing pickle files")):
            try:
                if os.path.exists(pickle_path):
                    output_dir = output_dirs[i]
                    self.process_pickle_file(pickle_path, output_dir, create_visualization)

                    # Create overlay visualization if requested
                    if create_visualization:
                        camera_path = os.path.join(dataset_root, camera_paths[i])
                        lidar_png_path = os.path.join(output_dir, f"{Path(pickle_path).stem}.png")
                        visualize_output_path = (
                            camera_paths[i].replace("/camera/", "/lidar_png_visualize/").replace(".png", "_overlay.png")
                        )
                        if output_root and output_root != dataset_root:
                            visualize_output_path = os.path.join(output_root, visualize_output_path)
                        elif not output_root:
                            # Keep as relative path when output_root is empty
                            pass
                        else:
                            visualize_output_path = os.path.join(dataset_root, visualize_output_path)

                        if os.path.exists(camera_path) and os.path.exists(lidar_png_path):
                            points3d_vis, _, camera_coord_vis = self.load_pickle_data(pickle_path)
                            self.create_overlay_visualization(
                                camera_path, lidar_png_path, visualize_output_path, points3d_vis, camera_coord_vis
                            )
                        else:
                            print(
                                f"Warning: Missing files for overlay - camera: {camera_path}, lidar: {lidar_png_path}"
                            )

                    processed_count += 1
                else:
                    print(f"Warning: Pickle file not found: {pickle_path}")
            except Exception as e:
                print(f"Error processing {pickle_path}: {e}")
                continue

        print(f"Processing complete. Processed {processed_count}/{len(pickle_paths)} files.")


def main():
    # Hardcoded paths for XOD dataset
    input_path = "xod_dataset/all.txt"
    dataset_root = "xod_dataset"
    output_root = None  # Will default to dataset_root
    create_visualization = False

    # Initialize projector
    projector = XODL2DProjector()

    # Print camera intrinsics info
    print("Using XOD L2D projection with pre-computed camera coordinates:")
    print(f"Output size: {projector.output_width}x{projector.output_height} (matches annotation dimensions)")
    for key, value in projector.camera_intrinsics.items():
        print(f"  {key}: {value}")
    print(f"LiDAR mean: {projector.lidar_mean}")
    print(f"LiDAR std: {projector.lidar_std}")
    print("Coordinate scaling: Camera 1 coords from 4240x2824 to 1363x768 output")
    print()

    # Check input type
    input_path = Path(input_path)
    if input_path.is_file():
        if input_path.suffix.lower() == ".txt":
            print(f"Processing from file list: {input_path}")
            projector.process_from_file_list(
                str(input_path),
                dataset_root=dataset_root,
                output_root=output_root,
                create_visualization=create_visualization,
            )
        else:
            if not output_root:
                output_root = dataset_root
            os.makedirs(output_root, exist_ok=True)
            print(f"Processing single pickle file: {input_path}")

            # Find corresponding camera image
            camera_path = None
            if create_visualization:
                # Assume camera images are in camera/ subdirectory at same level as pkl/
                pickle_dir = input_path.parent
                dataset_dir = pickle_dir.parent
                camera_dir = dataset_dir / "camera"
                pickle_name = input_path.stem
                camera_file = camera_dir / f"{pickle_name}.png"
                if camera_file.exists():
                    camera_path = str(camera_file)
                    print(f"Found camera image: {camera_path}")
                else:
                    print(f"Warning: Camera image not found at {camera_file}, visualization will be skipped")

            projector.process_pickle_file(str(input_path), output_root, create_visualization, camera_path)
    else:
        # Process directory
        print(f"Processing directory: {input_path}")
        projector.process_directory(str(input_path), create_visualization=create_visualization)


if __name__ == "__main__":
    main()
