from typing import List, Optional
from dataclasses import dataclass
from .handpose import HandPose


@dataclass
class TimedHandPose:
    pose: HandPose
    start_time: float  # seconds
    end_time: float    # seconds


class HandPoseSequence:
    def __init__(self, timed_poses: List[TimedHandPose]):
        # poses are sorted by start time
        self.sequence = sorted(timed_poses, key=lambda x: x.start_time)

    def get_pose_at_time(self, timestamp: float) -> Optional[HandPose]:
        """
        Returns the pose active at a given timestamp.
        """
        for timed_pose in self.sequence:
            if timed_pose.start_time <= timestamp < timed_pose.end_time:
                return timed_pose.pose
        return None

    def get_all_timestamps(self) -> List[float]:
        """
        Returns the list of start_times for all poses.
        """
        return [tp.start_time for tp in self.sequence]

    def get_pose_by_index(self, index: int) -> HandPose:
        return self.sequence[index].pose

    @property
    def current_pose(self) -> Optional[HandPose]:
        """
        Returns the current pose based on the latest available timestamp.
        Defaults to the last one if no time-tracking is active.
        """
        if self.sequence:
            return self.sequence[-1].pose
        return None

    def __getitem__(self, index: int) -> TimedHandPose:
        return self.sequence[index]

    def __len__(self):
        return len(self.sequence)

    def __str__(self):
        return f"<HandPoseSequence with {len(self.sequence)} poses>"
