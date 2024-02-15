

import cv2
import contextlib

def _create_video_writer(cap, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    return out_video_writer

@contextlib.contextmanager
def video_capture(path):
    """Context manager for video capture."""
    cap = cv2.VideoCapture(path)
    try:
        yield cap
    finally:
        cap.release()
        
@contextlib.contextmanager
def video_writer(*args, **kwargs):
    writer = _create_video_writer(*args, **kwargs)
    try:
        yield writer
    finally:
        writer.release()

if __name__ == "__main__":
    pass