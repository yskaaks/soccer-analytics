
from tqdm import tqdm
import cv2
import pandas as pd

def create_progress_bar(cap, desc="Processing Video"):
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=total_frames, desc=desc)
    
    return progress_bar


def update_tracks_csv(tracked_objects, csv_file_path, frame_number):
    # Attempt to load the existing CSV file into a DataFrame
    try:
        df = pd.read_csv(csv_file_path, index_col='ID')
    except FileNotFoundError:
        # If the file does not exist, initialize an empty DataFrame with ID as index
        df = pd.DataFrame(columns=['ID']).set_index('ID')

    # Column name for the current frame
    frame_col_name = f'Frame_{frame_number}'

    # Update DataFrame with new data
    for tracked_obj in tracked_objects:
        box_id = tracked_obj.id
        
        x1, y1 = tracked_obj.estimate[0]
        x2, y2 = tracked_obj.estimate[1]
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        
        box_xyxy_list = str([int(x1), int(y1), int(x2), int(y2)])
        
        if box_id in df.index:
            df.at[box_id, frame_col_name] = box_xyxy_list
        else:
            # Add a new row for new IDs
            df.loc[box_id, frame_col_name] = box_xyxy_list

    # Fill NaN values with an empty string or a placeholder if necessary
    df.fillna('', inplace=True)

    # Save the updated DataFrame back to CSV
    df.to_csv(csv_file_path)

def find_continuous_sequences(csv_file_path):
    df = pd.read_csv(csv_file_path, index_col='ID')
    sequences = {}

    for box_id in df.index:
        sequences[box_id] = []
        current_sequence = []
        last_frame = None

        for col in df.columns:
            if not pd.isna(df.at[box_id, col]):
                frame_number = int(col.split('_')[1])
                if last_frame is None or frame_number == last_frame + 1:
                    current_sequence.append((frame_number, eval(df.at[box_id, col])))
                else:
                    sequences[box_id].append(current_sequence)
                    current_sequence = [(frame_number, eval(df.at[box_id, col]))]
                last_frame = frame_number
            elif current_sequence:
                sequences[box_id].append(current_sequence)
                current_sequence = []
                last_frame = None
        if current_sequence:
            sequences[box_id].append(current_sequence)
    
    return sequences

def _calculate_surface_area(bbox):
    _, _, w, h = bbox  # Assuming bbox format is [x1, y1, x2, y2]
    return w * h

def update_sequences_to_uniform_size(sequences):
    updated_sequences = {}

    for box_id, seq_list in sequences.items():
        updated_sequences[box_id] = []
        for seq in seq_list:
            # Find bbox with biggest area        
            max_area_bbox = max(seq, key=lambda x: _calculate_surface_area(x[1]))
            target_width = max_area_bbox[1][2] - max_area_bbox[1][0]
            target_height = max_area_bbox[1][3] - max_area_bbox[1][1]

            updated_seq = []
            for frame_num, bbox in seq:
                center_x = bbox[0] + (bbox[2] - bbox[0]) / 2
                center_y = bbox[1] + (bbox[3] - bbox[1]) / 2

                # Update each bbox in the sequence
                new_bbox = [
                    int(center_x - target_width / 2),
                    int(center_y - target_height / 2),
                    int(center_x + target_width / 2),
                    int(center_y + target_height / 2)
                ]
                updated_seq.append((frame_num, new_bbox))
            updated_sequences[box_id].append(updated_seq)

    return updated_sequences


if __name__ == "__main__":
    pass