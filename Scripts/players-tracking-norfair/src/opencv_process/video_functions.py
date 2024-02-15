
import cv2
import os
from tqdm import tqdm

def create_video_writer(cap, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    return out_video_writer

def save_sequences_as_videos(sequences, video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    base_video_name = os.path.splitext(os.path.basename(video_path))[0]  # Extract base name without extension

    for box_id, seq_list in tqdm(sequences.items(), desc="Saving sequences"):
        sequence_counter = 1
        for seq in seq_list:
            start_frame, end_frame = seq[0][0], seq[-1][0]
            # Ensure each sequence has a unique name, especially if an ID has multiple sequences
            video_name = f"{base_video_name}_{box_id}_sequence{sequence_counter}_{start_frame}_{end_frame}.mp4"
            output_video_path = os.path.join(output_folder, video_name)

            width = seq[0][1][2] - seq[0][1][0]  # Assuming all bboxes in a sequence have the same size
            height = seq[0][1][3] - seq[0][1][1]
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            for frame_num, bbox in seq:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
                ret, frame = cap.read()
                if ret:
                    crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    out.write(crop)
            out.release()
            sequence_counter += 1  # Increment to ensure unique naming for each sequence

    cap.release()

if __name__ == "__main__":
    pass