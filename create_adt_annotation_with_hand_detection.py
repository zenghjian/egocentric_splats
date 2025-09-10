#!/usr/bin/env python3
"""
Create ADT annotation template with automatic person/hand identification.
This script integrates skeleton data to determine which person's hand is manipulating objects.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
import argparse
import traceback
from projectaria_tools.projects.adt import (
    AriaDigitalTwinDataProvider,
    AriaDigitalTwinDataPathsProvider,
    AriaDigitalTwinSkeletonProvider,
    MotionType
)
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.sensor_data import TimeQueryOptions

# Post-processing functions for concurrent sequences
def find_interaction_overlaps(interactions1: List[Dict], interactions2: List[Dict], 
                             duration_tolerance: float = 0.5) -> List[Tuple[int, int, float]]:
    """
    Find matching interactions between two concurrent annotation files based on:
    1. Same object_id  
    2. Similar position in sequence (concurrent recording preserves order)
    3. Similar duration characteristics
    """
    overlaps = []
    
    # Group interactions by object_id to maintain order
    obj_groups1 = {}
    obj_groups2 = {}
    
    for i, int1 in enumerate(interactions1):
        obj_id = int1['object_id']
        if obj_id not in obj_groups1:
            obj_groups1[obj_id] = []
        obj_groups1[obj_id].append((i, int1))
    
    for j, int2 in enumerate(interactions2):
        obj_id = int2['object_id']
        if obj_id not in obj_groups2:
            obj_groups2[obj_id] = []
        obj_groups2[obj_id].append((j, int2))
    
    # Match interactions by object and sequence position
    for obj_id in obj_groups1.keys():
        if obj_id not in obj_groups2:
            continue
            
        group1 = obj_groups1[obj_id]
        group2 = obj_groups2[obj_id]
        
        # Match interactions at same positions within each object group
        min_len = min(len(group1), len(group2))
        
        for k in range(min_len):
            idx1, int1 = group1[k]
            idx2, int2 = group2[k]
            
            # Calculate similarity score based on duration and motion characteristics
            dur1 = int1['duration_ms']
            dur2 = int2['duration_ms']
            
            path1 = int1['motion_stats']['path_length_m']
            path2 = int2['motion_stats']['path_length_m']
            
            # Duration similarity (closer durations = higher score)
            duration_ratio = min(dur1, dur2) / max(dur1, dur2) if max(dur1, dur2) > 0 else 0
            
            # Path similarity (closer path lengths = higher score)  
            path_ratio = min(path1, path2) / max(path1, path2) if max(path1, path2) > 0 else 0
            
            # Combined similarity score
            similarity = (duration_ratio + path_ratio) / 2
            
            # Consider it a match if similarity is high enough
            if similarity > duration_tolerance:
                overlaps.append((idx1, idx2, similarity))
    
    return overlaps

def postprocess_concurrent_annotations(file1_path: Path, file2_path: Path) -> Tuple[Dict, Dict]:
    """
    Post-process two concurrent annotation files to remove duplicate interactions.
    Keep the interaction with the smaller hand distance for each overlapping pair.
    """
    
    # Load annotation files
    with open(file1_path, 'r') as f:
        data1 = json.load(f)
    with open(file2_path, 'r') as f:
        data2 = json.load(f)
    
    interactions1 = data1['interactions']
    interactions2 = data2['interactions']
    
    print(f"File 1: {len(interactions1)} interactions")
    print(f"File 2: {len(interactions2)} interactions")
    
    # Find matching interactions
    overlaps = find_interaction_overlaps(interactions1, interactions2)
    print(f"Found {len(overlaps)} matching interaction pairs")
    
    # Determine which interactions to remove based on hand distance
    remove_from_file1 = set()
    remove_from_file2 = set()
    
    for idx1, idx2, similarity_score in overlaps:
        int1 = interactions1[idx1]
        int2 = interactions2[idx2]
        
        dist1 = int1['avg_hand_distance_m']
        dist2 = int2['avg_hand_distance_m']
        
        print(f"  {int1['object_name']}: {int1['skeleton_name']} ({dist1:.3f}m) vs {int2['skeleton_name']} ({dist2:.3f}m)")
        
        if dist1 < dist2:
            remove_from_file2.add(idx2)
            print(f"    → Keeping {int1['skeleton_name']} (closer)")
        else:
            remove_from_file1.add(idx1)
            print(f"    → Keeping {int2['skeleton_name']} (closer)")
    
    # Create filtered interactions and update interaction_ids
    filtered_interactions1 = []
    for new_idx, (old_idx, interaction) in enumerate((i, int) for i, int in enumerate(interactions1) if i not in remove_from_file1):
        updated_interaction = interaction.copy()
        id_parts = interaction['interaction_id'].rsplit('_', 1)
        if len(id_parts) == 2:
            base_id, _ = id_parts
            updated_interaction['interaction_id'] = f"{base_id}_{new_idx}"
        filtered_interactions1.append(updated_interaction)
    
    filtered_interactions2 = []
    for new_idx, (old_idx, interaction) in enumerate((i, int) for i, int in enumerate(interactions2) if i not in remove_from_file2):
        updated_interaction = interaction.copy()
        id_parts = interaction['interaction_id'].rsplit('_', 1)
        if len(id_parts) == 2:
            base_id, _ = id_parts
            updated_interaction['interaction_id'] = f"{base_id}_{new_idx}"
        filtered_interactions2.append(updated_interaction)
    
    # Update data
    data1['interactions'] = filtered_interactions1
    data2['interactions'] = filtered_interactions2
    
    # Add processing metadata
    data1['postprocessing'] = {
        'concurrent_processing': True,
        'removed_interactions': len(remove_from_file1),
        'original_count': len(interactions1),
        'final_count': len(filtered_interactions1)
    }
    
    data2['postprocessing'] = {
        'concurrent_processing': True,
        'removed_interactions': len(remove_from_file2),
        'original_count': len(interactions2), 
        'final_count': len(filtered_interactions2)
    }
    
    return data1, data2

# Joint indices for hands/wrists in the ADT skeleton
# Based on the actual AriaDigitalTwinSkeletonProvider.get_joint_labels():
# Left arm: 5 (LShoulder), 6 (LUArm), 7 (LFArm), 8 (LHand), 9-23 (left fingers)
# Right arm: 24 (RShoulder), 25 (RUArm), 26 (RFArm), 27 (RHand), 28-42 (right fingers)
JOINT_INDICES = {
    'left_wrist': 8,     # LHand
    'left_elbow': 7,     # LFArm
    'right_wrist': 27,   # RHand
    'right_elbow': 26,   # RFArm
    'left_hand': 15,     # LMiddle1 (middle of left hand)
    'right_hand': 34     # RMiddle1 (middle of right hand)
}

def initialize_adt_provider(sequence_path: Path) -> AriaDigitalTwinDataProvider:
    """Initialize ADT data provider for a sequence."""
    paths_provider = AriaDigitalTwinDataPathsProvider(str(sequence_path))
    data_paths = paths_provider.get_datapaths(False)
    
    if data_paths is None:
        raise ValueError(f"Failed to get data paths from {sequence_path}")
    
    return AriaDigitalTwinDataProvider(data_paths)

def load_skeleton_provider(sequence_path: Path) -> Tuple[Optional[AriaDigitalTwinSkeletonProvider], Dict]:
    """
    Load skeleton provider and association data.
    
    Returns:
        Tuple of (skeleton_provider, skeleton_association)
        Returns (None, {}) if skeleton data is not available
    """
    # Check for skeleton files
    skeleton_file = sequence_path / "Skeleton_T.json"
    assoc_file = sequence_path / "skeleton_aria_association.json"
    
    if not skeleton_file.exists():
        print("   - No skeleton data found (Skeleton_T.json missing)")
        return None, {}
    
    # Initialize skeleton provider
    try:
        skeleton_provider = AriaDigitalTwinSkeletonProvider(str(skeleton_file))
        print(f"   - Initialized AriaDigitalTwinSkeletonProvider")
        
        # Get joint labels for reference (this is a class method)
        joint_labels = AriaDigitalTwinSkeletonProvider.get_joint_labels()
        print(f"   - Loaded {len(joint_labels)} joint labels")
        
    except Exception as e:
        print(f"   - Failed to initialize skeleton provider: {e}")
        return None, {}
    
    # Load skeleton-device association if available
    skeleton_assoc = {}
    if assoc_file.exists():
        with open(assoc_file, 'r') as f:
            skeleton_assoc = json.load(f)
            print(f"   - Loaded skeleton association data")
    
    return skeleton_provider, skeleton_assoc

def get_skeleton_for_device(skeleton_assoc: Dict, device_serial: str) -> Optional[str]:
    """Get skeleton name for a specific device serial."""
    for meta in skeleton_assoc['SkeletonMetadata']:
        if device_serial in meta['AssociatedDeviceSerial']:
            return meta['SkeletonName']
    return None

def get_dynamic_objects(adt_provider: AriaDigitalTwinDataProvider) -> List[Tuple[int, str, str]]:
    """
    Get dynamic objects from the ADT provider following GIMO_ADT logic.
    
    Returns:
        List of (object_id, object_name, motion_type) tuples
    """
    dynamic_objects = []
    all_objects = []
    
    # Get all instance IDs
    instance_ids = adt_provider.get_instance_ids()
    print(f"   - Found {len(instance_ids)} total instances in sequence")
    
    # Check each instance to find dynamic objects
    for instance_id in instance_ids:
        try:
            instance_info = adt_provider.get_instance_info_by_id(instance_id)
            
            # Store all objects for fallback
            all_objects.append((instance_id, instance_info.name, str(instance_info.motion_type)))
            
            # Check if this is a dynamic object
            if instance_info.motion_type == MotionType.DYNAMIC:
                dynamic_objects.append((instance_id, instance_info.name, "DYNAMIC"))
        except Exception as e:
            print(f"   - Error processing instance {instance_id}: {e}")
            continue
    
    # If no dynamic objects found, fall back to using all objects
    if not dynamic_objects:
        print("   - No dynamic objects found, using all objects as fallback")
        dynamic_objects = all_objects[:50]  # Limit to 50 objects
    
    print(f"   - Found {len(dynamic_objects)} objects to track")
    return dynamic_objects

def get_hand_positions_at_timestamp(skeleton_provider: Optional[AriaDigitalTwinSkeletonProvider],
                                   target_timestamp_ns: int) -> Dict:
    """
    Get hand positions at a specific timestamp using the skeleton provider API.
    
    Returns:
        Dict of joint_name -> position (numpy array)
    """
    if not skeleton_provider:
        return {}
    
    try:
        # Query skeleton frame at the timestamp
        skeleton_frame_with_dt = skeleton_provider.get_skeleton_by_timestamp_ns(
            target_timestamp_ns,
            TimeQueryOptions.CLOSEST
        )
        
        if not skeleton_frame_with_dt.is_valid():
            return {}
        
        # Get the actual skeleton data
        skeleton_frame = skeleton_frame_with_dt.data()
        joints = np.array(skeleton_frame.joints)
        
        # Extract hand/wrist positions
        hand_positions = {}
        for joint_name, joint_idx in JOINT_INDICES.items():
            if joint_idx < len(joints):
                hand_positions[joint_name] = joints[joint_idx]
        
        return hand_positions
        
    except Exception as e:
        # Silently fail - this happens frequently for timestamps outside skeleton range
        return {}

def calculate_hand_object_proximity(object_position: np.ndarray, 
                                   hand_positions: Dict) -> Tuple[str, float, str]:
    """
    Calculate which hand is closest to the object.
    
    Returns:
        Tuple of (hand_side, distance, interaction_type)
    """
    if not hand_positions:
        return "unknown", -1.0, "none"
    
    distances = {}
    
    # Calculate distances for each available hand position
    for joint_name in ['left_wrist', 'right_wrist', 'left_hand', 'right_hand']:
        if joint_name in hand_positions:
            distance = np.linalg.norm(object_position - hand_positions[joint_name])
            distances[joint_name] = distance
    
    if not distances:
        return "unknown", -1.0, "none"
    
    # Find closest hand
    closest_joint = min(distances, key=distances.get)
    # Debug print removed - was printing for every frame causing output flood
    closest_distance = distances[closest_joint]
    
    # Determine hand side
    if 'left' in closest_joint:
        hand_side = 'left'
    elif 'right' in closest_joint:
        hand_side = 'right'
    else:
        hand_side = 'unknown'
    
    # Determine interaction type based on distance
    if closest_distance < 0.15:  # Within 15cm
        interaction_type = "direct_manipulation"
    elif closest_distance < 0.30:  # Within 30cm
        interaction_type = "reaching"
    elif closest_distance < 0.50:  # Within 50cm
        interaction_type = "nearby"
    else:
        interaction_type = "distant"
    
    return hand_side, closest_distance, interaction_type

def track_object_with_hand_detection(adt_provider: AriaDigitalTwinDataProvider,
                                    object_id: int,
                                    timestamps: List[int],
                                    skeleton_provider: Optional[AriaDigitalTwinSkeletonProvider],
                                    skeleton_name: str,
                                    skip_frames: int = 10,
                                    pose_cache: Optional[Dict] = None) -> Tuple[List, List, List]:
    """
    Track object motion and detect hand interactions using pre-indexed skeleton.
    Uses pose caching to filter out static poses and reduce redundant data.
    
    Returns:
        Tuple of (poses, tracked_timestamps, hand_interactions)
    """
    poses = []
    tracked_timestamps = []
    hand_interactions = []
    
    for i in range(0, len(timestamps), skip_frames):
        ts = timestamps[i]
        
        try:
            # Get 3D bounding box at this timestamp
            bbox3d_with_dt = adt_provider.get_object_3d_boundingboxes_by_timestamp_ns(ts)
            
            if not bbox3d_with_dt.is_valid():
                continue
            
            bboxes3d = bbox3d_with_dt.data()
            
            if object_id not in bboxes3d:
                continue
            
            bbox = bboxes3d[object_id]
            
            # Check if object has actually moved using pose cache
            should_log_pose = True
            if pose_cache is not None:
                instance_info = adt_provider.get_instance_info_by_id(object_id)
                obj_name = instance_info.name
                
                if obj_name in pose_cache:
                    # Get current and cached poses
                    current_pose = bbox.transform_scene_object
                    old_pose = pose_cache[obj_name]
                    
                    # Calculate rotation and translation deltas
                    current_pose_R = current_pose.rotation().to_matrix()
                    current_pose_t = current_pose.translation()
                    old_pose_R = old_pose.rotation().to_matrix()
                    old_pose_t = old_pose.translation()
                    
                    norm_delta_R = np.linalg.norm(current_pose_R - old_pose_R, ord=2)
                    norm_delta_t = np.linalg.norm(current_pose_t - old_pose_t, ord=2)
                    
                    # Only log if movement exceeds thresholds
                    if norm_delta_R < 0.02 and norm_delta_t < 0.015:
                        should_log_pose = False
                    else:
                        # Update cache with new pose
                        pose_cache[obj_name] = current_pose
                else:
                    # First time seeing this object, add to cache
                    pose_cache[obj_name] = bbox.transform_scene_object
            
            # Only track this pose if object has moved
            if should_log_pose:
                # Extract position (geometric center)
                center = bbox.transform_scene_object.translation()[0]
                aabb = bbox.aabb
                height = aabb[3] - aabb[2]
                
                pos = np.array([center[0], center[1] + height/2.0, center[2]])
                
                poses.append(pos.tolist())
                tracked_timestamps.append(ts)
            
            # Get hand positions at this timestamp using skeleton provider
            if skeleton_provider:
                hand_positions = get_hand_positions_at_timestamp(
                    skeleton_provider, ts
                )
                
                # Calculate hand-object proximity
                hand_side, distance, interaction_type = calculate_hand_object_proximity(
                    pos, hand_positions
                )
                
                hand_interactions.append({
                    'timestamp_ns': ts,
                    'hand_side': hand_side,
                    'distance_m': distance,
                    'interaction_type': interaction_type,
                    'skeleton_name': skeleton_name
                })
            else:
                hand_interactions.append({
                    'timestamp_ns': ts,
                    'hand_side': 'unknown',
                    'distance_m': -1.0,
                    'interaction_type': 'none',
                    'skeleton_name': 'unknown'
                })
            
        except Exception as e:
            continue
    
    return poses, tracked_timestamps, hand_interactions

def detect_motion_segments_with_hands(poses: List[List[float]],
                                     timestamps: List[int],
                                     hand_interactions: List[Dict],
                                     min_segment_frames: int = 5) -> List[Dict]:
    """
    Create a single complete segment for the entire object tracking period.
    The motion filtering is already handled by the pose cache in track_object_with_hand_detection.
    This function creates one segment that spans the entire duration where the object was tracked.
    """
    if len(poses) < min_segment_frames:
        return []
    
    # Create one complete segment from all tracked data
    segment_poses = np.array(poses)
    # Calculate total path length (sum of distances between consecutive positions)
    path_length = np.sum(np.linalg.norm(np.diff(segment_poses, axis=0), axis=1))
    
    # Calculate displacement (straight-line distance from start to end)
    displacement = np.linalg.norm(segment_poses[-1] - segment_poses[0])
    
    # Analyze hand usage across the entire tracking period
    hand_stats = analyze_segment_hand_usage(hand_interactions)
    
    # Create single segment encompassing all tracked frames
    segment = {
        'start_timestamp_ns': timestamps[0],
        'end_timestamp_ns': timestamps[-1],
        'duration_ms': (timestamps[-1] - timestamps[0]) / 1e6,
        'num_frames': len(poses),
        'path_length_m': path_length,
        'displacement_m': displacement,
        'primary_hand': hand_stats['primary_hand'],
        'hand_confidence': hand_stats['confidence'],
        'avg_hand_distance_m': hand_stats['avg_distance'],
        'skeleton_name': hand_stats['skeleton_name'],
        'interaction_type': hand_stats['primary_interaction_type']
    }
    
    # Return single segment as a list
    return [segment]

def analyze_segment_hand_usage(hand_interactions: List[Dict]) -> Dict:
    """Analyze which hand was primarily used in a segment."""
    if not hand_interactions:
        return {
            'primary_hand': 'unknown',
            'confidence': 0.0,
            'avg_distance': -1.0,
            'skeleton_name': 'unknown',
            'primary_interaction_type': 'none'
        }
    
    # Count hand usage
    hand_counts = {'left': 0, 'right': 0, 'unknown': 0}
    distances = []
    interaction_types = []
    skeleton_names = []
    
    for interaction in hand_interactions:
        hand_counts[interaction.get('hand_side', 'unknown')] += 1
        
        dist = interaction.get('distance_m', -1.0)
        if dist >= 0:
            distances.append(dist)
        
        interaction_types.append(interaction.get('interaction_type', 'none'))
        skeleton_names.append(interaction.get('skeleton_name', 'unknown'))
    
    # Determine primary hand
    primary_hand = max(hand_counts, key=hand_counts.get)
    total_interactions = sum(hand_counts.values())
    confidence = hand_counts[primary_hand] / total_interactions if total_interactions > 0 else 0.0
    
    # Calculate average distance
    avg_distance = np.mean(distances) if distances else -1.0
    
    # Most common interaction type
    from collections import Counter
    interaction_counter = Counter(interaction_types)
    primary_interaction_type = interaction_counter.most_common(1)[0][0] if interaction_counter else 'none'
    
    # Most common skeleton
    skeleton_counter = Counter(skeleton_names)
    primary_skeleton = skeleton_counter.most_common(1)[0][0] if skeleton_counter else 'unknown'
    
    return {
        'primary_hand': primary_hand,
        'confidence': confidence,
        'avg_distance': avg_distance,
        'skeleton_name': primary_skeleton,
        'primary_interaction_type': primary_interaction_type
    }

def create_enhanced_annotation_template(sequence_name: str,
                                       all_interactions: List[Dict],
                                       device_serial: str) -> Dict:
    """Create annotation template with hand/person information."""
    template = {
        'sequence_name': sequence_name,
        'device_serial': device_serial,
        'annotation_date': datetime.now().isoformat(),
        'annotator': 'manual_with_hand_detection',
        'annotation_type': 'action_labels_with_hands',
        'source': 'ADT_provider_with_skeleton',
        'dataset_type': 'adt',
        'interactions': []
    }
    
    for idx, interaction in enumerate(all_interactions):
        annotation_entry = {
            'interaction_id': f"{sequence_name}_{interaction['object_id']}_{idx}",
            'object_id': interaction['object_id'],
            'object_name': interaction['object_name'],
            'motion_type': interaction['motion_type'],
            
            # Person/device information
            'person_device': device_serial,
            'skeleton_name': interaction.get('skeleton_name', 'unknown'),
            
            # Hand information (auto-detected)
            'detected_hand': interaction.get('primary_hand', 'unknown'),
            'hand_confidence': interaction.get('hand_confidence', 0.0),
            'avg_hand_distance_m': interaction.get('avg_hand_distance_m', -1.0),
            'detected_interaction_type': interaction.get('interaction_type', 'none'),
            
            # Temporal information
            'start_timestamp_ns': int(interaction['start_timestamp_ns']),
            'end_timestamp_ns': int(interaction['end_timestamp_ns']),
            'duration_ms': interaction['duration_ms'],
            'num_frames': interaction['num_frames'],
            
            # Motion statistics
            'motion_stats': {
                'path_length_m': interaction['path_length_m'],
                'displacement_m': interaction['displacement_m']
            },
            
            # Manual annotation fields (redundant fields removed)
            'action_label': "",
            'action_verb': "",
            'interaction_phase': "",
            'tool_used': "",
            'location': "",
            'goal': "",
            'notes': ""
            # Removed redundant fields:
            # - 'hand_used' (use 'detected_hand' instead)  
            # - 'confidence' (use 'hand_confidence' instead)
        }
        
        template['interactions'].append(annotation_entry)
    
    return template

def main():
    parser = argparse.ArgumentParser(description='Create ADT annotations with hand detection')
    parser.add_argument('--data_path', type=str,
                       default='/usr/stud/zehu/project/GIMO_ADT/data_sample',
                       help='Path to data sample directory')
    parser.add_argument('--output_dir', type=str,
                       default='adt_annotations_with_hands',
                       help='Output directory')
    # Motion threshold removed - now using pose cache thresholds (0.02 rotation, 0.015m translation)
    parser.add_argument('--min_motion_path', type=float, default=0.1,
                       help='Minimum path length in meters')
    args = parser.parse_args()
    
    base_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Check if base_path is a sequence directory itself or contains sequences
    if (base_path / "metadata.json").exists() or (base_path / "gt-metadata.json").exists():
        # Single sequence directory
        sequence_dirs = [base_path]
        print(f"Processing single sequence: {base_path.name}")
    else:
        # Parent directory with multiple sequences
        # Only include directories that have the required metadata files
        sequence_dirs = [
            d for d in base_path.iterdir() 
            if d.is_dir() and (
                (d / "metadata.json").exists() or 
                (d / "gt-metadata.json").exists()
            )
        ]
        print(f"Found {len(sequence_dirs)} valid sequences in {base_path}")
    
    for sequence_path in sequence_dirs:
        seq_name = sequence_path.name
        print(f"\n{'='*70}")
        print(f"Processing: {seq_name}")
        print(f"{'='*70}")
        
        try:
            # Extract device serial from sequence name
            if 'M1292' in seq_name:
                device_serial = '1WM103600M1292'
            elif '71292' in seq_name:
                device_serial = '1WM10360071292'
            else:
                device_serial = 'unknown'
            
            # Initialize ADT provider
            print("\n1. Initializing ADT provider...")
            adt_provider = initialize_adt_provider(sequence_path)
            
            # Load skeleton provider
            print("\n2. Loading skeleton data...")
            skeleton_provider, skeleton_assoc = load_skeleton_provider(sequence_path)
            
            # Get timestamps
            rgb_stream_id = StreamId("214-1")
            all_timestamps = adt_provider.get_aria_device_capture_timestamps_ns(rgb_stream_id)
            start_time = adt_provider.get_start_time_ns()
            end_time = adt_provider.get_end_time_ns()
            timestamps = [ts for ts in all_timestamps if start_time <= ts <= end_time]
            print(f"   - Found {len(timestamps)} valid timestamps")
            
            # Get skeleton name for this device
            print("\n3. Setting up skeleton tracking...")
            skeleton_name = None
            
            if skeleton_assoc and skeleton_provider:
                skeleton_name = get_skeleton_for_device(skeleton_assoc, device_serial)
                if skeleton_name:
                    print(f"   - Using skeleton: {skeleton_name}")
                else:
                    print(f"   - No skeleton found for device {device_serial}")
            
            if not skeleton_provider:
                print("   - No skeleton provider available, will track motion without hand detection")
            
            # Get dynamic objects
            print("\n4. Identifying dynamic objects...")
            dynamic_objects = get_dynamic_objects(adt_provider)
            
            # Track each object with hand detection
            print("\n5. Tracking objects with hand detection...")
            all_interactions = []
            
            # Initialize pose cache for tracking object movement
            # The pose cache will automatically filter out objects that don't move
            dynamic_obj_pose_cache = {}
            dynamic_obj_moved = set()
            
            print("   - Tracking dynamic objects with motion filtering...")
            
            # Track dynamic objects with pose caching for efficient motion detection
            for obj_id, obj_name, motion_type in dynamic_objects[:100]:  # Limit to top 100 objects
                # Track object with hand detection and motion filtering via pose cache
                poses, tracked_timestamps, hand_interactions = track_object_with_hand_detection(
                    adt_provider, obj_id, timestamps,
                    skeleton_provider, skeleton_name if skeleton_name else "unknown",
                    skip_frames=10,  # Sample every 10 frames
                    pose_cache=dynamic_obj_pose_cache  # Use pose cache for motion filtering
                )
                
                if len(poses) < 2:
                    continue
                
                # Create single segment for this object
                segments = detect_motion_segments_with_hands(
                    poses, tracked_timestamps, hand_interactions
                )
                
                # Check if segment meets minimum criteria
                if segments and segments[0]['path_length_m'] >= args.min_motion_path:
                    segment = segments[0]
                    
                    # Add object metadata
                    segment['object_id'] = obj_id
                    segment['object_name'] = obj_name
                    segment['motion_type'] = motion_type
                    
                    # Track which objects actually moved
                    dynamic_obj_moved.add(obj_name)
                    
                    print(f"   - {obj_name} (ID: {obj_id}): tracked for {segment['duration_ms']:.0f}ms, "
                          f"path: {segment['path_length_m']:.2f}m")
                    
                    # Print hand detection info if available
                    if segment['primary_hand'] != 'unknown':
                        print(f"     → {segment['primary_hand']} hand detected "
                              f"(confidence: {segment['hand_confidence']:.2f}, "
                              f"avg distance: {segment['avg_hand_distance_m']:.3f}m)")
                    
                    all_interactions.append(segment)
            
            print(f"   - Total segments found: {len(all_interactions)}")
            if dynamic_obj_moved:
                print(f"   - Objects with significant motion: {', '.join(list(dynamic_obj_moved)[:5])}{'...' if len(dynamic_obj_moved) > 5 else ''}")
            
            # Create enhanced template
            print("\n6. Creating enhanced annotation template...")
            template = create_enhanced_annotation_template(seq_name, all_interactions, device_serial)
            
            # Save annotation file
            annotation_file = output_dir / f"{seq_name}.json"
            with open(annotation_file, 'w') as f:
                json.dump(template, f, indent=2)
            print(f"   - Saved: {annotation_file}")
            
            # Print sample with hand info
            if all_interactions:
                print("\n7. Sample interactions with hand detection:")
                for inter in all_interactions[:3]:
                    print(f"   - {inter['object_name']}: {inter['duration_ms']:.0f}ms")
                    print(f"     Hand: {inter.get('primary_hand', 'unknown')} "
                          f"(confidence: {inter.get('hand_confidence', 0):.2f})")
                    print(f"     Distance: {inter.get('avg_hand_distance_m', -1.0):.3f}m")
                    print(f"     Type: {inter.get('interaction_type', 'none')}")
            
        except Exception as e:
            print(f"ERROR processing {seq_name}: {e}")
            traceback.print_exc()
    
    # Post-process concurrent annotations if multiple sequences were processed
    print(f"\n{'='*70}")
    print(f"Initial processing completed! Output saved to: {output_dir.absolute()}")
    
    # Check if we have concurrent sequences (M1292 and 71292)
    annotation_files = list(output_dir.glob("*.json"))
    concurrent_files = []
    
    for file in annotation_files:
        if 'M1292' in file.name or '71292' in file.name:
            concurrent_files.append(file)
    
    if len(concurrent_files) == 2:
        print(f"\n{'='*70}")
        print("POST-PROCESSING CONCURRENT SEQUENCES")
        print(f"{'='*70}")
        print("Detected concurrent sequences. Running post-processing to remove duplicate interactions...")
        
        try:
            # Sort files to ensure consistent order
            file1 = next(f for f in concurrent_files if 'M1292' in f.name)
            file2 = next(f for f in concurrent_files if '71292' in f.name)
            
            # Run post-processing
            processed_data1, processed_data2 = postprocess_concurrent_annotations(file1, file2)
            
            # Save optimized versions
            optimized_file1 = output_dir / f"{file1.stem}_optimized.json"
            optimized_file2 = output_dir / f"{file2.stem}_optimized.json"
            
            with open(optimized_file1, 'w') as f:
                json.dump(processed_data1, f, indent=2)
            with open(optimized_file2, 'w') as f:
                json.dump(processed_data2, f, indent=2)
            
            print(f"\nOptimized files saved:")
            print(f"  - {optimized_file1}")
            print(f"  - {optimized_file2}")
            
            # Print final summary
            total1 = len(processed_data1['interactions'])
            total2 = len(processed_data2['interactions'])
            skeleton1 = processed_data1['interactions'][0]['skeleton_name'] if processed_data1['interactions'] else 'Unknown'
            skeleton2 = processed_data2['interactions'][0]['skeleton_name'] if processed_data2['interactions'] else 'Unknown'
            
            print(f"\nFinal Results:")
            print(f"  {skeleton1}: {total1} unique interactions")
            print(f"  {skeleton2}: {total2} unique interactions")
            print(f"  Total: {total1 + total2} interactions (duplicates removed)")
            
        except Exception as e:
            print(f"Post-processing failed: {e}")
            print("Original files are still available.")
    
    print(f"\n{'='*70}")
    print(f"All processing completed!")

if __name__ == "__main__":
    main()