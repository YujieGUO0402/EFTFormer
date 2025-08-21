from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, BinaryIO, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import io
import os

import numpy as np
import numpy.typing as npt
from PIL import Image
from pyquaternion import Quaternion
from sympy import Li


from nuplan.planning.simulation.occupancy_map.abstract_occupancy_map import OccupancyMap
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.database.maps_db.gpkg_mapsdb import MAP_LOCATIONS
from nuplan.database.utils.pointclouds.lidar import LidarPointCloud

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)

NAVSIM_INTERVAL_LENGTH: float = 0.5
OPENSCENE_DATA_ROOT = os.environ.get("OPENSCENE_DATA_ROOT")
NUPLAN_MAPS_ROOT = os.environ.get("NUPLAN_MAPS_ROOT")


@dataclass
class Camera:
    """Camera dataclass for image and parameters."""

    image: Optional[npt.NDArray[np.float32]] = None

    sensor2lidar_rotation: Optional[npt.NDArray[np.float32]] = None
    sensor2lidar_translation: Optional[npt.NDArray[np.float32]] = None
    intrinsics: Optional[npt.NDArray[np.float32]] = None
    distortion: Optional[npt.NDArray[np.float32]] = None


@dataclass
class Cameras:
    """Multi-camera dataclass."""

    cam_f0: Camera
    cam_l0: Camera
    cam_l1: Camera
    cam_l2: Camera
    cam_r0: Camera
    cam_r1: Camera
    cam_r2: Camera
    cam_b0: Camera

    @classmethod
    def from_camera_dict(
        cls,
        sensor_blobs_path: Path,
        camera_dict: Dict[str, Any],
        sensor_names: List[str],
    ) -> Cameras:
        """
        Load camera dataclass from dictionary.
        :param sensor_blobs_path: root directory of sensor data.
        :param camera_dict: dictionary containing camera specifications.
        :param sensor_names: list of camera identifiers to include.
        :return: Cameras dataclass.
        """
        ###
        # missing_images_log = "/media/yujie/data/E2EAD/endtoenddriving_test/missing_images.txt"
        data_dict: Dict[str, Camera] = {}
        for camera_name in camera_dict.keys():
            camera_identifier = camera_name.lower()
            if camera_identifier in sensor_names:
                image_path = sensor_blobs_path / camera_dict[camera_name]["data_path"]
                data_dict[camera_identifier] = Camera(
                    image=np.array(Image.open(image_path)),
                    sensor2lidar_rotation=camera_dict[camera_name]["sensor2lidar_rotation"],
                    sensor2lidar_translation=camera_dict[camera_name]["sensor2lidar_translation"],
                    intrinsics=camera_dict[camera_name]["cam_intrinsic"],
                    distortion=camera_dict[camera_name]["distortion"],
                )


                # ###
                # try:
                #     image = np.array(Image.open(image_path))
                # except FileNotFoundError:
                #     with open(missing_images_log, "a") as f:
                #         f.write(f"{image_path}\n")
                #     # 如果图片缺失，则用空 Camera 占位
                #     data_dict[camera_identifier] = Camera()
                #     continue
                # data_dict[camera_identifier] = Camera(
                #     image=image,
                #     sensor2lidar_rotation=camera_dict[camera_name]["sensor2lidar_rotation"],
                #     sensor2lidar_translation=camera_dict[camera_name]["sensor2lidar_translation"],
                #     intrinsics=camera_dict[camera_name]["cam_intrinsic"],
                #     distortion=camera_dict[camera_name]["distortion"],
                # )


            else:
                data_dict[camera_identifier] = Camera()  # empty camera

        return Cameras(
            cam_f0=data_dict["cam_f0"],
            cam_l0=data_dict["cam_l0"],
            cam_l1=data_dict["cam_l1"],
            cam_l2=data_dict["cam_l2"],
            cam_r0=data_dict["cam_r0"],
            cam_r1=data_dict["cam_r1"],
            cam_r2=data_dict["cam_r2"],
            cam_b0=data_dict["cam_b0"],
        )


@dataclass
class Lidar:
    """Lidar point cloud dataclass."""

    # NOTE:
    # merged lidar point cloud as (6,n) float32 array with n points
    # first axis: (x, y, z, intensity, ring, lidar_id), see LidarIndex
    lidar_pc: Optional[npt.NDArray[np.float32]] = None

    @staticmethod
    def _load_bytes(lidar_path: Path) -> BinaryIO:
        """Helper static method to load lidar point cloud stream."""
        with open(lidar_path, "rb") as fp:
            return io.BytesIO(fp.read())

    @classmethod
    def from_paths(cls, sensor_blobs_path: Path, lidar_path: Path, sensor_names: List[str]) -> Lidar:
        """
        Loads lidar point cloud dataclass in log loading.
        :param sensor_blobs_path: root directory to sensor data
        :param lidar_path: relative lidar path from logs.
        :param sensor_names: list of sensor identifiers to load`
        :return: lidar point cloud dataclass
        """

        # NOTE: this could be extended to load specific LiDARs in the merged pc
        if "lidar_pc" in sensor_names:
            global_lidar_path = sensor_blobs_path / lidar_path
            lidar_pc = LidarPointCloud.from_buffer(cls._load_bytes(global_lidar_path), "pcd").points
            return Lidar(lidar_pc)
        return Lidar()  # empty lidar

        ###
        # missing_lidar_log = "/media/yujie/data/E2EAD/endtoenddriving_test/missing_lidar.txt"
        # if "lidar_pc" in sensor_names:
        #     global_lidar_path = sensor_blobs_path / lidar_path
            
        #     # Try loading the lidar file, skip if missing
        #     try:
        #         lidar_pc = LidarPointCloud.from_buffer(cls._load_bytes(global_lidar_path), "pcd").points
        #         return Lidar(lidar_pc)
        #     except FileNotFoundError:
        #         print(f"[Warning] Lidar file missing: {global_lidar_path}")
                
        #         # Log the missing LiDAR path to a file
        #         with open(missing_lidar_log, "a") as log_file:
        #             log_file.write(str(global_lidar_path) + "\n")
                
        #         return Lidar()  # Return an empty Lidar object if the file is missing
        
        return Lidar() 


@dataclass
class EgoStatus:
    """Ego vehicle status dataclass."""

    ego_pose: npt.NDArray[np.float64]
    ego_velocity: npt.NDArray[np.float32]
    ego_acceleration: npt.NDArray[np.float32]
    driving_command: npt.NDArray[np.int]
    in_global_frame: bool = False  # False for AgentInput

@dataclass
class Occ_gt:
    """Occupancy ground truth dataclass."""

    # NOTE: occupancy grid as a numpy array
    # The shape of the occupancy grid could be [H, W], where H and W are the height and width of the grid
    occ_grid: Optional[np.ndarray] = None

    def __post_init__(self):
        """在初始化后验证数据"""
        if self.occ_grid is not None:
            print(f"Occ_gt initialized with grid shape: {self.occ_grid.shape}")
            print(f"Grid dtype: {self.occ_grid.dtype}")
            print(f"Grid min/max values: {self.occ_grid.min()}/{self.occ_grid.max()}")
            if np.isnan(self.occ_grid).any():
                print("Warning: Grid contains NaN values!")
            if np.isinf(self.occ_grid).any():
                print("Warning: Grid contains Inf values!")

    @staticmethod
    def _load_bytes(occ_gt_path: Path) -> io.BytesIO:
        """Helper static method to load occupancy grid stream."""
        print(f"Loading occupancy grid from: {occ_gt_path}")
        try:
            with open(occ_gt_path, "rb") as fp:
                return io.BytesIO(fp.read())
        except Exception as e:
            print(f"Error loading occupancy grid: {e}")
            raise

    @classmethod
    def from_paths(cls, sensor_blobs_path: Path, occ_gt_path: Path) -> Occ_gt:
        """
        Loads occupancy ground truth dataclass in log loading.
        :param sensor_blobs_path: root directory to sensor data
        :param occ_gt_path: relative occupancy ground truth path from logs.
        :return: Occupancy ground truth dataclass
        """
        print(f"Loading Occ_gt from paths:")
        print(f"sensor_blobs_path: {sensor_blobs_path}")
        print(f"occ_gt_path: {occ_gt_path}")
        
        # Compute the full path to the occupancy ground truth file
        root_path = sensor_blobs_path.parents[2]
        global_occ_gt_path = root_path / occ_gt_path
        print(f"Global occ_gt path: {global_occ_gt_path}")

        try:
            # Load the .npy file containing the occupancy grid
            occ_grid = np.load(global_occ_gt_path)
            print(f"Successfully loaded occupancy grid with shape: {occ_grid.shape}")
            
            # Return the occupancy ground truth object
            return Occ_gt(occ_grid)
        except Exception as e:
            print(f"Error in from_paths: {e}")
            raise

@dataclass
class AgentInput:
    """Dataclass for agent inputs with current and past ego statuses and sensors."""

    ego_statuses: List[EgoStatus]
    cameras: List[Cameras]
    lidars: List[Lidar]

    @classmethod
    def from_scene_dict_list(
        cls,
        scene_dict_list: List[Dict],
        sensor_blobs_path: Path,
        num_history_frames: int,
        sensor_config: SensorConfig,
    ) -> AgentInput:
        """
        Load agent input from scene dictionary.
        :param scene_dict_list: list of scene frames (in logs).
        :param sensor_blobs_path: root directory of sensor data
        :param num_history_frames: number of agent input frames
        :param sensor_config: sensor config dataclass
        :return: agent input dataclass
        """
        assert len(scene_dict_list) > 0, "Scene list is empty!"

        global_ego_poses = []
        for frame_idx in range(num_history_frames):
            ego_translation = scene_dict_list[frame_idx]["ego2global_translation"]
            ego_quaternion = Quaternion(*scene_dict_list[frame_idx]["ego2global_rotation"])
            global_ego_pose = np.array(
                [ego_translation[0], ego_translation[1], ego_quaternion.yaw_pitch_roll[0]],
                dtype=np.float64,
            )
            global_ego_poses.append(global_ego_pose)

        local_ego_poses = convert_absolute_to_relative_se2_array(
            StateSE2(*global_ego_poses[-1]), np.array(global_ego_poses, dtype=np.float64)
        )

        ego_statuses: List[EgoStatus] = []
        cameras: List[EgoStatus] = []
        lidars: List[Lidar] = []
    
        for frame_idx in range(num_history_frames):

            ego_dynamic_state = scene_dict_list[frame_idx]["ego_dynamic_state"]
            ego_status = EgoStatus(
                ego_pose=np.array(local_ego_poses[frame_idx], dtype=np.float32),
                ego_velocity=np.array(ego_dynamic_state[:2], dtype=np.float32),
                ego_acceleration=np.array(ego_dynamic_state[2:], dtype=np.float32),
                driving_command=scene_dict_list[frame_idx]["driving_command"],
            )
            ego_statuses.append(ego_status)

            sensor_names = sensor_config.get_sensors_at_iteration(frame_idx)
            cameras.append(
                Cameras.from_camera_dict(
                    sensor_blobs_path=sensor_blobs_path,
                    camera_dict=scene_dict_list[frame_idx]["cams"],
                    sensor_names=sensor_names,
                )
            )

            lidars.append(
                Lidar.from_paths(
                    sensor_blobs_path=sensor_blobs_path,
                    lidar_path=Path(scene_dict_list[frame_idx]["lidar_path"]),
                    sensor_names=sensor_names,
                )
            )

        return AgentInput(ego_statuses, cameras, lidars)

@dataclass
class OccTarget:
    """Dataclass for targets with current and past occs."""
    occs: List[Occ_gt]

    def __post_init__(self):
        """在初始化后验证数据"""
        print(f"OccTarget initialized with {len(self.occs)} frames")
        for i, occ in enumerate(self.occs):
            print(f"Frame {i}:")
            if occ.occ_grid is not None:
                print(f"  Grid shape: {occ.occ_grid.shape}")
                print(f"  Grid dtype: {occ.occ_grid.dtype}")
                print(f"  Grid min/max values: {occ.occ_grid.min()}/{occ.occ_grid.max()}")

    @classmethod
    def from_scene_dict_list(
        cls,
        scene_dict_list: List[Dict],
        sensor_blobs_path: Path,
        num_history_frames: int,
    ) -> OccTarget:
        """
        Load occ from scene dictionary.
        :param scene_dict_list: list of scene frames (in logs).
        :param sensor_blobs_path: root directory of sensor data
        :param num_history_frames: number of agent input frames
        :return: occ dataclass
        """
        print(f"Loading OccTarget from scene dict:")
        print(f"Number of frames: {len(scene_dict_list)}")
        print(f"Number of history frames: {num_history_frames}")
        print(f"Sensor blobs path: {sensor_blobs_path}")

        assert len(scene_dict_list) > 0, "Scene list is empty!"

        occs: List[Occ_gt] = []
    
        for frame_idx in range(num_history_frames):
            print(f"\nProcessing frame {frame_idx}:")
            try:
                occ_gt_path = Path(scene_dict_list[frame_idx]["occ_gt_final_path"])
                print(f"Loading occ_gt from: {occ_gt_path}")
                
                occ = Occ_gt.from_paths(
                    sensor_blobs_path=sensor_blobs_path,
                    occ_gt_path=occ_gt_path,
                )
                occs.append(occ)
                print(f"Successfully loaded frame {frame_idx}")
            except Exception as e:
                print(f"Error loading frame {frame_idx}: {e}")
                raise

        return OccTarget(occs)

@dataclass
class Annotations:
    """Dataclass of annotations (e.g. bounding boxes) per frame."""

    boxes: npt.NDArray[np.float32]
    names: List[str]
    velocity_3d: npt.NDArray[np.float32]
    instance_tokens: List[str]
    track_tokens: List[str]

    def __post_init__(self):
        annotation_lengths: Dict[str, int] = {
            attribute_name: len(attribute) for attribute_name, attribute in vars(self).items()
        }
        assert (
            len(set(annotation_lengths.values())) == 1
        ), f"Annotations expects all attributes to have equal length, but got {annotation_lengths}"


@dataclass
class Trajectory:
    """Trajectory dataclass in NAVSIM."""

    poses: npt.NDArray[np.float32]  # local coordinates
    trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

    def __post_init__(self):
        assert self.poses.ndim == 2, "Trajectory poses should have two dimensions for samples and poses."
        assert (
            self.poses.shape[0] == self.trajectory_sampling.num_poses
        ), "Trajectory poses and sampling have unequal number of poses."
        assert self.poses.shape[1] == 3, "Trajectory requires (x, y, heading) at last dim."


@dataclass
class SceneMetadata:
    """Dataclass of scene metadata (e.g. location) per scene."""

    log_name: str
    scene_token: str
    map_name: str
    initial_token: str

    num_history_frames: int
    num_future_frames: int


@dataclass
class Frame:
    """Frame dataclass with privileged information."""

    token: str
    timestamp: int
    roadblock_ids: List[str]
    traffic_lights: List[Tuple[str, bool]]
    annotations: Annotations

    ego_status: EgoStatus
    lidar: Lidar
    cameras: Cameras
    #occs: Occ_gt


@dataclass
class Scene:
    """Scene dataclass defining a single sample in NAVSIM."""

    # Ground truth information
    scene_metadata: SceneMetadata
    map_api: AbstractMap
    frames: List[Frame]

    def get_future_trajectory(self, num_trajectory_frames: Optional[int] = None) -> Trajectory:
        """
        Extracts future trajectory of the human operator in local coordinates (ie. ego rear-axle).
        :param num_trajectory_frames: optional number frames to extract poses, defaults to None
        :return: trajectory dataclass
        """

        if num_trajectory_frames is None:
            num_trajectory_frames = self.scene_metadata.num_future_frames

        start_frame_idx = self.scene_metadata.num_history_frames - 1

        global_ego_poses = []
        for frame_idx in range(start_frame_idx, start_frame_idx + num_trajectory_frames + 1):
            global_ego_poses.append(self.frames[frame_idx].ego_status.ego_pose)

        local_ego_poses = convert_absolute_to_relative_se2_array(
            StateSE2(*global_ego_poses[0]), np.array(global_ego_poses[1:], dtype=np.float64)
        )

        return Trajectory(
            local_ego_poses,
            TrajectorySampling(
                num_poses=len(local_ego_poses),
                interval_length=NAVSIM_INTERVAL_LENGTH,
            ),
        )

    def get_history_trajectory(self, num_trajectory_frames: Optional[int] = None) -> Trajectory:
        """
        Extracts past trajectory of ego vehicles in local coordinates (ie. ego rear-axle).
        :param num_trajectory_frames: optional number frames to extract poses, defaults to None
        :return: trajectory dataclass
        """

        if num_trajectory_frames is None:
            num_trajectory_frames = self.scene_metadata.num_history_frames

        global_ego_poses = []
        for frame_idx in range(num_trajectory_frames):
            global_ego_poses.append(self.frames[frame_idx].ego_status.ego_pose)

        origin = StateSE2(*global_ego_poses[-1])
        local_ego_poses = convert_absolute_to_relative_se2_array(origin, np.array(global_ego_poses, dtype=np.float64))

        return Trajectory(
            local_ego_poses,
            TrajectorySampling(
                num_poses=len(local_ego_poses),
                interval_length=NAVSIM_INTERVAL_LENGTH,
            ),
        )

    def get_agent_input(self) -> AgentInput:
        """
        Extracts agents input dataclass (without privileged information) from scene.
        :return: agent input dataclass
        """

        local_ego_poses = self.get_history_trajectory().poses
        ego_statuses: List[EgoStatus] = []
        cameras: List[Cameras] = []
        lidars: List[Lidar] = []

        for frame_idx in range(self.scene_metadata.num_history_frames):
            frame_ego_status = self.frames[frame_idx].ego_status

            ego_statuses.append(
                EgoStatus(
                    ego_pose=local_ego_poses[frame_idx],
                    ego_velocity=frame_ego_status.ego_velocity,
                    ego_acceleration=frame_ego_status.ego_acceleration,
                    driving_command=frame_ego_status.driving_command,
                )
            )
            cameras.append(self.frames[frame_idx].cameras)
            lidars.append(self.frames[frame_idx].lidar)
            

        return AgentInput(ego_statuses, cameras, lidars)

    # def get_occ(self) -> OccTarget:
    #     """
    #     Extracts the occupancy grid of the current frame (used as target).
    #     :param frame_idx: The index of the frame to retrieve occ_gt.
    #     :return: The occ_gt for the current frame.
    #     """
    #     occs: List[Occ_gt] = []

    #     for frame_idx in range(self.scene_metadata.num_history_frames):
    #         occs.append(self.frames[frame_idx].occs)
            
    #     return OccTarget(occs)

    @classmethod
    def _build_map_api(cls, map_name: str) -> AbstractMap:
        """Helper classmethod to load map api from name."""
        assert map_name in MAP_LOCATIONS, f"The map name {map_name} is invalid, must be in {MAP_LOCATIONS}"
        return get_maps_api(NUPLAN_MAPS_ROOT, "nuplan-maps-v1.0", map_name)

    @classmethod
    def _build_annotations(cls, scene_frame: Dict) -> Annotations:
        """Helper classmethod to load annotation dataclass from logs."""
        return Annotations(
            boxes=scene_frame["anns"]["gt_boxes"],
            names=scene_frame["anns"]["gt_names"],
            velocity_3d=scene_frame["anns"]["gt_velocity_3d"],
            instance_tokens=scene_frame["anns"]["instance_tokens"],
            track_tokens=scene_frame["anns"]["track_tokens"],
        )

    @classmethod
    def _build_ego_status(cls, scene_frame: Dict) -> EgoStatus:
        """Helper classmethod to load ego status dataclass from logs."""
        ego_translation = scene_frame["ego2global_translation"]
        ego_quaternion = Quaternion(*scene_frame["ego2global_rotation"])
        global_ego_pose = np.array(
            [ego_translation[0], ego_translation[1], ego_quaternion.yaw_pitch_roll[0]],
            dtype=np.float64,
        )
        ego_dynamic_state = scene_frame["ego_dynamic_state"]
        return EgoStatus(
            ego_pose=global_ego_pose,
            ego_velocity=np.array(ego_dynamic_state[:2], dtype=np.float32),
            ego_acceleration=np.array(ego_dynamic_state[2:], dtype=np.float32),
            driving_command=scene_frame["driving_command"],
            in_global_frame=True,
        )

    @classmethod
    def from_scene_dict_list(
        cls,
        scene_dict_list: List[Dict],
        sensor_blobs_path: Path,
        num_history_frames: int,
        num_future_frames: int,
        sensor_config: SensorConfig,
    ) -> Scene:
        """
        Load scene dataclass from scene dictionary list (for log loading).
        :param scene_dict_list: list of scene frames (in logs)
        :param sensor_blobs_path: root directory of sensor data
        :param num_history_frames: number of past and current frames to load
        :param num_future_frames: number of future frames to load
        :param sensor_config: sensor config dataclass
        :return: scene dataclass
        """
        assert len(scene_dict_list) >= 0, "Scene list is empty!"
        scene_metadata = SceneMetadata(
            log_name=scene_dict_list[num_history_frames - 1]["log_name"],
            scene_token=scene_dict_list[num_history_frames - 1]["scene_token"],
            map_name=scene_dict_list[num_history_frames - 1]["map_location"],
            initial_token=scene_dict_list[num_history_frames - 1]["token"],
            num_history_frames=num_history_frames,
            num_future_frames=num_future_frames,
        )
        map_api = cls._build_map_api(scene_metadata.map_name)

        frames: List[Frame] = []
        for frame_idx in range(len(scene_dict_list)):
            global_ego_status = cls._build_ego_status(scene_dict_list[frame_idx])
            annotations = cls._build_annotations(scene_dict_list[frame_idx])

            sensor_names = sensor_config.get_sensors_at_iteration(frame_idx)

            cameras = Cameras.from_camera_dict(
                sensor_blobs_path=sensor_blobs_path,
                camera_dict=scene_dict_list[frame_idx]["cams"],
                sensor_names=sensor_names,
            )

            lidar = Lidar.from_paths(
                sensor_blobs_path=sensor_blobs_path,
                lidar_path=Path(scene_dict_list[frame_idx]["lidar_path"]),
                sensor_names=sensor_names,
            )

            # occ_gt_path = scene_dict_list[frame_idx]["occ_gt_final_path"]  # Load Occ_gt

            # if occ_gt_path is None:
            #     print(f"[Warning] occ_gt_final_path is None in scene with token: {scene_dict_list[frame_idx]['scene_token']}")
            #     with open("missing_occ_gt_scenes.txt", "a") as f:
            #         f.write(f"{scene_dict_list[frame_idx]['scene_token']}\n")
            #     return None  # 或者 raise 一个特定的异常，例如 SkipSceneException()

            # occs = Occ_gt.from_paths(
            #     sensor_blobs_path=sensor_blobs_path,
            #     occ_gt_path=Path(occ_gt_path),
            # )

            frame = Frame(
                token=scene_dict_list[frame_idx]["token"],
                timestamp=scene_dict_list[frame_idx]["timestamp"],
                roadblock_ids=scene_dict_list[frame_idx]["roadblock_ids"],
                traffic_lights=scene_dict_list[frame_idx]["traffic_lights"],
                annotations=annotations,
                ego_status=global_ego_status,
                lidar=lidar,
                cameras=cameras,
                #occs=occs,
            )
            frames.append(frame)

        return Scene(scene_metadata=scene_metadata, map_api=map_api, frames=frames)


@dataclass
class SceneFilter:
    """Scene filtering configuration for scene loading."""

    num_history_frames: int = 4
    num_future_frames: int = 8
    frame_interval: Optional[int] = None
    has_route: bool = True

    max_scenes: Optional[int] = None
    log_names: Optional[List[str]] = None
    tokens: Optional[List[str]] = None
    # TODO: expand filter options

    def __post_init__(self):

        if self.frame_interval is None:
            self.frame_interval = self.num_frames

        assert self.num_history_frames >= 1, "SceneFilter: num_history_frames must greater equal one."
        assert self.num_future_frames >= 0, "SceneFilter: num_future_frames must greater equal zero."
        assert self.frame_interval >= 1, "SceneFilter: frame_interval must greater equal one."

    @property
    def num_frames(self) -> int:
        """
        :return: total number for frames for scenes to extract.
        """
        return self.num_history_frames + self.num_future_frames


@dataclass
class SensorConfig:
    """Configuration dataclass of agent sensors for memory management."""

    # Config values of sensors are either
    # - bool: Whether to load history or not
    # - List[int]: For loading specific history steps
    cam_f0: Union[bool, List[int]]
    cam_l0: Union[bool, List[int]]
    cam_l1: Union[bool, List[int]]
    cam_l2: Union[bool, List[int]]
    cam_r0: Union[bool, List[int]]
    cam_r1: Union[bool, List[int]]
    cam_r2: Union[bool, List[int]]
    cam_b0: Union[bool, List[int]]
    lidar_pc: Union[bool, List[int]]

    def get_sensors_at_iteration(self, iteration: int) -> List[str]:
        """
        Creates a list of sensor identifiers given iteration.
        :param iteration: integer indicating the history iteration.
        :return: list of sensor identifiers to load.
        """
        sensors_at_iteration: List[str] = []
        for sensor_name, sensor_include in asdict(self).items():
            if isinstance(sensor_include, bool) and sensor_include:
                sensors_at_iteration.append(sensor_name)
            elif isinstance(sensor_include, list) and iteration in sensor_include:
                sensors_at_iteration.append(sensor_name)
        return sensors_at_iteration

    @classmethod
    def build_all_sensors(cls, include: Union[bool, List[int]] = True) -> SensorConfig:
        """
        Classmethod to load all sensors with the same specification.
        :param include: boolean or integers for sensors to include, defaults to True
        :return: sensor configuration dataclass
        """
        return SensorConfig(
            cam_f0=include,
            cam_l0=include,
            cam_l1=include,
            cam_l2=include,
            cam_r0=include,
            cam_r1=include,
            cam_r2=include,
            cam_b0=include,
            lidar_pc=include,
        )

    @classmethod
    def build_no_sensors(cls) -> SensorConfig:
        """
        Classmethod to load no sensors.
        :return: sensor configuration dataclass
        """
        return cls.build_all_sensors(include=False)


@dataclass
class PDMResults:
    """Helper dataclass to record PDM results."""

    no_at_fault_collisions: float
    drivable_area_compliance: float

    ego_progress: float
    time_to_collision_within_bound: float
    comfort: float
    driving_direction_compliance: float

    score: float
