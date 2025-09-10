#!/usr/bin/env python3
"""
完整演示 AriaDigitalTwinSkeletonProvider API 的所有功能。
基于实际的 projectaria_tools API 文档。
"""

import numpy as np
from pathlib import Path
import argparse
from typing import List, Optional
import json

from projectaria_tools.projects.adt import (
    AriaDigitalTwinSkeletonProvider,
    AriaDigitalTwinDataProvider,
    AriaDigitalTwinDataPathsProvider,
)
from projectaria_tools.core.sensor_data import TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId


def print_section(title: str):
    """打印格式化的章节标题"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def demonstrate_skeleton_provider(sequence_path: Path):
    """
    演示 AriaDigitalTwinSkeletonProvider 的所有功能
    """
    
    # 1. 初始化
    print_section("1. 初始化 AriaDigitalTwinSkeletonProvider")
    
    skeleton_file = sequence_path / "Skeleton_T.json"
    if not skeleton_file.exists():
        print(f"❌ 未找到骨架文件: {skeleton_file}")
        return
    
    try:
        skeleton_provider = AriaDigitalTwinSkeletonProvider(str(skeleton_file))
        print(f"✅ 成功初始化 AriaDigitalTwinSkeletonProvider")
        print(f"   文件路径: {skeleton_file}")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return
    
    # 2. 获取关节标签 (这是类方法，不是实例方法!)
    print_section("2. 关节标签 (Joint Labels)")
    
    joint_labels = None
    try:
        # 注意：这是类方法，从类调用而不是从实例调用
        joint_labels = AriaDigitalTwinSkeletonProvider.get_joint_labels()
        print(f"✅ 获取到 {len(joint_labels)} 个关节标签")
        print("\n前10个关节标签:")
        for i, label in enumerate(joint_labels[:10]):
            print(f"   关节 {i:2d}: {label}")
        
        print("\n所有关节标签:")
        print(f"   {joint_labels}")
    except Exception as e:
        print(f"❌ 获取关节标签失败: {e}")
        joint_labels = None
    
    # 3. 获取关节连接 (这也是类方法!)
    print_section("3. 关节连接 (Joint Connections)")
    
    try:
        # 注意：这也是类方法
        joint_connections = AriaDigitalTwinSkeletonProvider.get_joint_connections()
        print(f"✅ 获取到 {len(joint_connections)} 个关节连接")
        print("\n前10个连接 (parent_id, child_id):")
        for i, (parent, child) in enumerate(joint_connections[:10]):
            if joint_labels and parent < len(joint_labels) and child < len(joint_labels):
                print(f"   {parent:2d} ({joint_labels[parent]:15s}) -> {child:2d} ({joint_labels[child]:15s})")
            else:
                print(f"   {parent:2d} -> {child:2d}")
    except Exception as e:
        print(f"❌ 获取关节连接失败: {e}")
        joint_connections = None
    
    # 4. 获取标记标签 (这也是类方法!)
    print_section("4. 标记标签 (Marker Labels)")
    
    try:
        # 注意：这也是类方法
        marker_labels = AriaDigitalTwinSkeletonProvider.get_marker_labels()
        print(f"✅ 获取到 {len(marker_labels)} 个标记标签")
        print("\n前10个标记标签:")
        for i, label in enumerate(marker_labels[:10]):
            print(f"   标记 {i:2d}: {label}")
    except Exception as e:
        print(f"❌ 获取标记标签失败: {e}")
        marker_labels = None
    
    # 5. 获取时间戳并查询骨架数据
    print_section("5. 查询骨架帧数据")
    
    # 需要从ADT数据提供器获取有效时间戳
    try:
        paths_provider = AriaDigitalTwinDataPathsProvider(str(sequence_path))
        data_paths = paths_provider.get_datapaths()
        adt_provider = AriaDigitalTwinDataProvider(data_paths)
        
        # 获取RGB相机的时间戳
        rgb_stream = StreamId("214-1")
        timestamps = adt_provider.get_aria_device_capture_timestamps_ns(rgb_stream)
        
        if timestamps and len(timestamps) > 0:
            print(f"✅ 获取到 {len(timestamps)} 个时间戳")
            
            # 测试不同的时间查询选项
            test_timestamps = [
                timestamps[0],      # 第一帧
                timestamps[len(timestamps)//2],  # 中间帧
                timestamps[-1]      # 最后一帧
            ]
            
            query_options = [
                (TimeQueryOptions.CLOSEST, "CLOSEST"),
                (TimeQueryOptions.BEFORE, "BEFORE"),
                (TimeQueryOptions.AFTER, "AFTER")
            ]
            
            for ts_idx, test_ts in enumerate(test_timestamps):
                print(f"\n测试时间戳 {ts_idx + 1}: {test_ts} ns ({test_ts/1e9:.3f} 秒)")
                
                for option, option_name in query_options:
                    try:
                        skeleton_frame_with_dt = skeleton_provider.get_skeleton_by_timestamp_ns(
                            test_ts, option
                        )
                        
                        if skeleton_frame_with_dt.is_valid():
                            print(f"\n  使用 {option_name} 查询:")
                            print(f"    ✅ 有效的骨架帧")
                            print(f"    时间偏移: {skeleton_frame_with_dt.dt_ns} ns")
                            
                            # 获取实际的骨架数据
                            skeleton_frame = skeleton_frame_with_dt.data()
                            
                            # 关节数据
                            joints = skeleton_frame.joints
                            print(f"    关节数量: {len(joints)}")
                            if len(joints) > 0:
                                print(f"    第一个关节位置: {joints[0]}")
                                print(f"    关节数据类型: {type(joints[0])}")
                                
                                # 计算骨架中心
                                joints_array = np.array(joints)
                                center = np.mean(joints_array, axis=0)
                                print(f"    骨架中心位置: {center}")
                                
                                # 计算骨架边界框
                                min_pos = np.min(joints_array, axis=0)
                                max_pos = np.max(joints_array, axis=0)
                                bbox_size = max_pos - min_pos
                                print(f"    骨架边界框大小: {bbox_size}")
                            
                            # 标记数据
                            markers = skeleton_frame.markers
                            print(f"    标记数量: {len(markers)}")
                            if len(markers) > 0:
                                print(f"    第一个标记位置: {markers[0]}")
                        else:
                            print(f"  使用 {option_name} 查询: ❌ 无效的骨架帧")
                            
                    except Exception as e:
                        print(f"  使用 {option_name} 查询失败: {e}")
                
                # 只测试第一个时间戳的详细信息
                if ts_idx == 0:
                    break
                    
        else:
            print("❌ 无法获取时间戳")
            
    except Exception as e:
        print(f"❌ 查询骨架帧失败: {e}")
    
    # 6. 演示完整的骨架跟踪
    print_section("6. 骨架跟踪演示")
    
    try:
        if timestamps and len(timestamps) > 10:
            # 采样一些帧
            sample_indices = np.linspace(0, len(timestamps)-1, 10, dtype=int)
            
            print(f"跟踪 {len(sample_indices)} 个采样帧的骨架运动:")
            
            positions = []
            valid_frames = 0
            
            for idx in sample_indices:
                ts = timestamps[idx]
                skeleton_frame_with_dt = skeleton_provider.get_skeleton_by_timestamp_ns(ts)
                
                if skeleton_frame_with_dt.is_valid():
                    skeleton_frame = skeleton_frame_with_dt.data()
                    joints = skeleton_frame.joints
                    
                    if len(joints) > 0:
                        # 计算骨架中心
                        center = np.mean(np.array(joints), axis=0)
                        positions.append(center)
                        valid_frames += 1
            
            print(f"  有效帧数: {valid_frames}/{len(sample_indices)}")
            
            if len(positions) > 1:
                positions = np.array(positions)
                
                # 计算运动统计
                total_displacement = np.linalg.norm(positions[-1] - positions[0])
                path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
                
                print(f"  总位移: {total_displacement:.3f} 米")
                print(f"  路径长度: {path_length:.3f} 米")
                print(f"  平均速度: {path_length / (timestamps[sample_indices[-1]] - timestamps[sample_indices[0]]) * 1e9:.3f} 米/秒")
                
    except Exception as e:
        print(f"❌ 骨架跟踪失败: {e}")
    
    # 7. 总结
    print_section("7. API 总结")
    
    print("""
AriaDigitalTwinSkeletonProvider 提供的功能:

1. AriaDigitalTwinSkeletonProvider.get_joint_labels() -> List[str]
   - 类方法！从类调用，不是从实例调用
   - 获取所有关节的名称标签（51个关节）
   - 第i个元素对应关节ID i

2. AriaDigitalTwinSkeletonProvider.get_joint_connections() -> List[Tuple[int, int]]
   - 类方法！从类调用，不是从实例调用
   - 获取关节之间的连接关系（20个连接）
   - 返回 (父关节ID, 子关节ID) 对的列表

3. AriaDigitalTwinSkeletonProvider.get_marker_labels() -> List[str]
   - 类方法！从类调用，不是从实例调用
   - 获取OptiTrack标记点的标签（57个标记）
   - 第i个元素对应标记ID i

4. skeleton_provider.get_skeleton_by_timestamp_ns(timestamp_ns, time_query_options)
   - 实例方法！需要先创建provider实例
   - 根据时间戳查询骨架帧
   - time_query_options: CLOSEST, BEFORE, 或 AFTER
   - 返回 SkeletonFrameWithDt 对象

5. SkeletonFrameWithDt:
   - is_valid(): 检查帧是否有效
   - dt_ns: 时间偏移（纳秒）
   - data(): 返回 SkeletonFrame 对象

6. SkeletonFrame:
   - joints: 关节位置列表 (3D坐标)
   - markers: 标记位置列表 (3D坐标)

使用场景:
- 人体姿态跟踪和分析
- 动作识别和行为分析
- 人机交互研究
- 第一人称视角的人体运动理解
    """)


def main():
    parser = argparse.ArgumentParser(
        description='演示 AriaDigitalTwinSkeletonProvider API'
    )
    parser.add_argument(
        '--sequence_path',
        type=str,
        default='./data/Apartment_release_golden_skeleton_seq100_10s_sample_M1292',
        help='ADT序列目录路径'
    )
    
    args = parser.parse_args()
    sequence_path = Path(args.sequence_path)
    
    if not sequence_path.exists():
        print(f"❌ 错误: 序列路径不存在: {sequence_path}")
        return
    
    print(f"📂 ADT序列路径: {sequence_path}")
    demonstrate_skeleton_provider(sequence_path)


if __name__ == "__main__":
    main()