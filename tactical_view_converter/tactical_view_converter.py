import os
import sys
import pathlib
import numpy as np
import cv2
from copy import deepcopy
from .homography import Homography

folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path,"../"))
from utils import get_foot_position,measure_distance

class TacticalViewConverter:
    def __init__(self, court_image_path):
        self.court_image_path = court_image_path
        self.width = 300
        self.height= 161

        self.actual_width_in_meters=28
        self.actual_height_in_meters=15 

        self.key_points = [
            # left edge
            (0,0),
            (0,int((0.91/self.actual_height_in_meters)*self.height)),
            (0,int((5.18/self.actual_height_in_meters)*self.height)),
            (0,int((10/self.actual_height_in_meters)*self.height)),
            (0,int((14.1/self.actual_height_in_meters)*self.height)),
            (0,int(self.height)),

            # Middle line
            (int(self.width/2),self.height),
            (int(self.width/2),0),
            
            # Left Free throw line
            (int((5.79/self.actual_width_in_meters)*self.width),int((5.18/self.actual_height_in_meters)*self.height)),
            (int((5.79/self.actual_width_in_meters)*self.width),int((10/self.actual_height_in_meters)*self.height)),

            # right edge
            (self.width,int(self.height)),
            (self.width,int((14.1/self.actual_height_in_meters)*self.height)),
            (self.width,int((10/self.actual_height_in_meters)*self.height)),
            (self.width,int((5.18/self.actual_height_in_meters)*self.height)),
            (self.width,int((0.91/self.actual_height_in_meters)*self.height)),
            (self.width,0),

            # Right Free throw line
            (int(((self.actual_width_in_meters-5.79)/self.actual_width_in_meters)*self.width),int((5.18/self.actual_height_in_meters)*self.height)),
            (int(((self.actual_width_in_meters-5.79)/self.actual_width_in_meters)*self.width),int((10/self.actual_height_in_meters)*self.height)),
        ]

    def validate_keypoints(self, keypoints_list):
        """
        Validates detected keypoints by comparing their proportional distances
        to the tactical view keypoints.

        Args:
            keypoints_list (List[List[Tuple[float, float]]]): A list containing keypoints for each frame.
                Each outer list represents a frame.
                Each inner list contains keypoints as (x, y) tuples.
                A keypoint of (0, 0) indicates that the keypoint is not detected for that frame.

        Returns:
            List[List[Tuple[float, float]]]: A list of validated keypoints for each frame.
                Frames with missing or invalid keypoints will have empty lists.
        """

        keypoints_list = deepcopy(keypoints_list)
        validated_list = []

        for frame_idx, frame_keypoints in enumerate(keypoints_list):
            if frame_keypoints is None:
                # Replace None frames with empty list
                validated_list.append([])
                continue

            # Convert Detectron2 Keypoints object to list if needed
            if hasattr(frame_keypoints, 'xy'):
                try:
                    frame_keypoints_xy = frame_keypoints.xy.tolist()[0]
                except Exception:
                    frame_keypoints_xy = []
            else:
                frame_keypoints_xy = frame_keypoints

            if not frame_keypoints_xy:
                validated_list.append([])
                continue

            # Copy so we can modify
            frame_kp = deepcopy(frame_keypoints_xy)

            # Get indices of detected keypoints (not (0, 0))
            detected_indices = [i for i, kp in enumerate(frame_kp) if kp[0] > 0 and kp[1] > 0]

            # Need at least 3 detected keypoints to validate proportions
            if len(detected_indices) < 3:
                validated_list.append(frame_kp)
                continue

            invalid_keypoints = []
            for i in detected_indices:
                if frame_kp[i][0] == 0 and frame_kp[i][1] == 0:
                    continue

                other_indices = [idx for idx in detected_indices if idx != i and idx not in invalid_keypoints]
                if len(other_indices) < 2:
                    continue
                j, k = other_indices[0], other_indices[1]

                d_ij = measure_distance(frame_kp[i], frame_kp[j])
                d_ik = measure_distance(frame_kp[i], frame_kp[k])
                t_ij = measure_distance(self.key_points[i], self.key_points[j])
                t_ik = measure_distance(self.key_points[i], self.key_points[k])

                if t_ij > 0 and t_ik > 0:
                    prop_detected = d_ij / d_ik if d_ik > 0 else float('inf')
                    prop_tactical = t_ij / t_ik if t_ik > 0 else float('inf')

                    error = abs((prop_detected - prop_tactical) / prop_tactical)

                    if error > 0.8:  # 80% error margin
                        frame_kp[i] = (0, 0)  # mark invalid
                        invalid_keypoints.append(i)

            validated_list.append(frame_kp)

        return validated_list


    def transform_players_to_tactical_view(self, keypoints_list, player_tracks):
            """
            Transform player positions from video frame coordinates to tactical view coordinates.
            
            Args:
                keypoints_list (list): List of detected court keypoints for each frame.
                player_tracks (list): List of dictionaries containing player tracking information for each frame,
                    where each dictionary maps player IDs to their bounding box coordinates.
            
            Returns:
                list: List of dictionaries where each dictionary maps player IDs to their (x, y) positions
                    in the tactical view coordinate system. The list index corresponds to the frame number.
            """
            tactical_player_positions = []
            
            for frame_idx, (frame_keypoints, frame_tracks) in enumerate(zip(keypoints_list, player_tracks)):
                # Initialize empty dictionary for this frame
                tactical_positions = {}

                if frame_keypoints is None or len(frame_keypoints) == 0:
                    tactical_player_positions.append({})
                    continue

                frame_keypoints_xy = frame_keypoints.xy.tolist()[0] if hasattr(frame_keypoints, 'xy') else frame_keypoints


                # Skip frames with insufficient keypoints
                if frame_keypoints is None or len(frame_keypoints) == 0:
                    tactical_player_positions.append(tactical_positions)
                    continue
                
                # Get detected keypoints for this frame
                detected_keypoints = frame_keypoints
                
                # Filter out undetected keypoints (those with coordinates (0,0))
                valid_indices = [i for i, kp in enumerate(detected_keypoints) if kp[0] > 0 and kp[1] > 0]
                
                # Need at least 4 points for a reliable homography
                if len(valid_indices) < 4:
                    tactical_player_positions.append(tactical_positions)
                    continue
                
                # Create source and target point arrays for homography
                source_points = np.array([detected_keypoints[i] for i in valid_indices], dtype=np.float32)
                target_points = np.array([self.key_points[i] for i in valid_indices], dtype=np.float32)
                
                try:
                    # Create homography transformer
                    homography = Homography(source_points, target_points)
                    
                    # Transform each player's position
                    for player_id, player_data in frame_tracks.items():
                        bbox = player_data["bbox"]
                        # Use bottom center of bounding box as player position
                        player_position = np.array([get_foot_position(bbox)])
                        # Transform to tactical view coordinates
                        tactical_position = homography.transform_points(player_position)

                        # If tactical position is not in the tactical view, skip
                        if tactical_position[0][0] < 0 or tactical_position[0][0] > self.width or tactical_position[0][1] < 0 or tactical_position[0][1] > self.height:
                            continue

                        tactical_positions[player_id] = tactical_position[0].tolist()
                        
                except (ValueError, cv2.error) as e:
                    # If homography fails, continue with empty dictionary
                    pass
                
                tactical_player_positions.append(tactical_positions)
            
            return tactical_player_positions

