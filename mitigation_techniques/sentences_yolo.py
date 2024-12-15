import os
import csv
import cv2
from sentences_scene_detect import get_extension

''' The datasets' videos has been sampled accordingly:
COCO:                           
-ActivityNetCap --> not sampled            
-Epic-Kitchens --> every 30th frame
-MSVD --> not sampled
-SumMe --> every 10th frame
-TVSum --> every 10th frame

OBJECTS365:
-ActivityNetCap --> every 10th frame
-Epic-Kitchens --> every 15th frame
-MSVD --> not sampled
-SumMe --> every 10th frame
-TVSum --> every 10th frame   

THIS PYTHON FILE IS FOR GENERATING SENTENCES FOR OBJECTS365 LABELS ONLY (FOR NOW) USING A RANKING SYSTEM
-takes top 12 detection with longest duration, later sorted by time again.'''


def frame_to_timestamp(frame, fps):
    return frame / fps

def rank_sentences(grouped_frames):
    # add duration for all
    for group in grouped_frames:
        group['duration'] = group['end'] - group['start']

    longest_groups = sorted(grouped_frames, key=lambda x: x['duration'], reverse=True)[:15] # take top 12
    # sort again by time
    ranked_sentences = sorted(longest_groups, key=lambda x: x['start'])

    return ranked_sentences


def write_sentences(dataset_label_p, dataset_n, output_p, video_p, sample_p, extension):
    output_file = os.path.join(output_p, f"{dataset_n}_sentences_coco_ranked.csv")
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Video name', 'Description'])

        for video_file in os.listdir(video_p):
            video_name = os.path.splitext(video_file)[0]
            print(f"Processing: {video_name}")
            label_file = f"{video_name}_labels.csv"
            label_path = os.path.join(dataset_label_p, label_file)

            if label_file.startswith('.~lock'):  # Skip any lock files
                print(f"Skipping lock file: {label_file}")
                continue

            # Read label file
            if os.path.exists(label_path):  # Ensure the label file exists
                with open(label_path, 'r') as csvfile:
                    reader = csv.DictReader(csvfile)
                    labels = [row for row in reader]
            else:
                print(f"Label file not found: {label_path}")
                continue

            # object x frame count
            frame_counts = {}
            for label in labels:
                frame = int(label['frame'])  # This is the sampled frame
                class_name = label['class']
                original_frame = frame * sample_p  # Convert to the original video frame

                # transform to original frame number
                if original_frame not in frame_counts:
                    frame_counts[original_frame] = {}
                if class_name not in frame_counts[original_frame]:
                    frame_counts[original_frame][class_name] = 0
                frame_counts[original_frame][class_name] += 1

            # smooth out small gaps
            smoothed_frame_counts = {}
            prev_frame = None
            prev_counts = {}

            for frame, counts in sorted(frame_counts.items()):
                if prev_frame is not None:
                    if frame - prev_frame == 1: # --> small gap, so merge counts
                        for class_name, count in counts.items():
                            prev_counts[class_name] = max(prev_counts.get(class_name, 0), count)
                    else:
                        # store and reset
                        smoothed_frame_counts[prev_frame] = prev_counts
                        prev_counts = counts.copy()
                else:
                    prev_counts = counts.copy() # for the first frame

                prev_frame = frame

            # add last frame
            if prev_frame is not None:
                smoothed_frame_counts[prev_frame] = prev_counts

            # group objects per frame
            grouped_labels = []
            current_group = None
            for frame, counts in smoothed_frame_counts.items():
                if current_group is None or counts != current_group['counts']:
                    if current_group is not None:
                        grouped_labels.append(current_group)
                    current_group = {'start': frame, 'end': frame, 'counts': counts}
                else:
                    current_group['end'] = frame
            if current_group is not None:
                grouped_labels.append(current_group)

            # RANKING THE GROUPS BY IT'S DURATION AND RETURNING TOP 10 SORTED BY START TIME?
            ranked_sentences = rank_sentences(grouped_labels)
            #ranked_sentences = grouped_labels
            # transform frames to timestamps and write the sentences
            video = cv2.VideoCapture(os.path.join(video_p, video_file))
            fps = video.get(cv2.CAP_PROP_FPS)
            output_sentences = []
            output_sentences.append(f"In the video {video_name} ")

            for group in ranked_sentences:
                start_timestamp = frame_to_timestamp(group['start'], fps)
                end_timestamp = frame_to_timestamp(group['end'], fps)
                sentence = ', '.join(f"{count} {class_name}" for class_name, count in group['counts'].items())
                sentence += f" appears from {start_timestamp:.2f} seconds to {end_timestamp:.2f} seconds."
                output_sentences.append(sentence)

            # Write the video name and description to the CSV
            description = ' '.join(output_sentences)
            writer.writerow([video_name + extension, description])


# ---------------------------------------------------------------------------

video_path = '/home/wikaaxx/Desktop/thesis/datasets' # put path for datasets' videos
labels_path = './data/YOLO/labels_coco'
output_path = './data/YOLO/sentences'
csv_output_file = os.path.join(output_path, '.csv')

for dataset_name in os.listdir(labels_path):
    dataset_label_path = os.path.join(labels_path, dataset_name)
    dataset_video_path = os.path.join(video_path, dataset_name, 'videos')  # datasets/dataset_name/videos
    if os.path.isdir(dataset_label_path):
        print(f"Dataset: {dataset_name}")
        # define sampling period
        if dataset_name == 'Epic-Kitchens':
            s = 15
        elif dataset_name == 'ActivityNetCap' or dataset_name == 'TVSum' or dataset_name == 'SumMe':
            s = 10
        else:
            s = 1  # like MSVD which hasn't been sampled
        # define extensions of the dataset's videos
        ext = get_extension(dataset_name)
        write_sentences(dataset_label_path, dataset_name, output_path, dataset_video_path, s, ext)
