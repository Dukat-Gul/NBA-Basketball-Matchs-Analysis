import supervision as sv
import numpy as np

class CourtKeypointDrawer:
    """
    A drawer class responsible for drawing court keypoints on a sequence of frames.
    """

    def __init__(self):
        self.keypoint_color = '#ff2c2c'

    def draw(self, frames, court_keypoints):
        """
        Draws court keypoints on a given list of frames.

        Args:
            frames (list): A list of frames (as NumPy arrays or image objects) on which to draw.
            court_keypoints (list): A corresponding list of lists where each sub-list contains
                the (x, y) coordinates of court keypoints for that frame.

        Returns:
            list: A list of frames with keypoints drawn on them.
        """
        vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex(self.keypoint_color),
            radius=8
        )

        vertex_label_annotator = sv.VertexLabelAnnotator(
            color=sv.Color.from_hex(self.keypoint_color),
            text_color=sv.Color.WHITE,
            text_scale=0.5,
            text_thickness=1
        )

        output_frames = []
        for index, frame in enumerate(frames):
            annotated_frame = frame.copy()
            keypoints = court_keypoints[index]

            # Skip empty keypoints
            if keypoints is None or len(keypoints) == 0:
                output_frames.append(annotated_frame)
                continue

            # Convert to numpy if it's a tensor
            if hasattr(keypoints, 'cpu'):  # PyTorch tensor
                keypoints_np = keypoints.cpu().numpy()
            else:  # Already a list or numpy array
                keypoints_np = np.array(keypoints)

            # Draw dots
            annotated_frame = vertex_annotator.annotate(
                scene=annotated_frame,
                key_points=keypoints_np
            )
            # Draw labels
            annotated_frame = vertex_label_annotator.annotate(
                scene=annotated_frame,
                key_points=keypoints_np
            )

            output_frames.append(annotated_frame)

        return output_frames
