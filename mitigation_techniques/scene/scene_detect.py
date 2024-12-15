import os
import json
from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg


def serialize_scenes(scene_list):
    # so it's just (start_time,end_time) per scene
    return [(scene[0].get_timecode(), scene[1].get_timecode()) for scene in scene_list]


def scene_detect_and_split(dataset_root, dataset, split_dir):
    video_folder = str(os.path.join(dataset_root, dataset, "videos"))  # videos of the dataset, datasets/dataset_name/videos
    split_folder = str(os.path.join(dataset_root, dataset, split_dir))  # for split videos, datasets/dataset_name/split_videos
    os.makedirs(split_folder, exist_ok=True)  # create if no split_videos folder

    for video_file in os.listdir(video_folder):
        if video_file.endswith((".avi", ".mp4", ".MP4")):
            video_path = os.path.join(video_folder, video_file)
            video_name_no_ext = os.path.splitext(os.path.basename(video_path))[0]
            print('    -video: '+str(video_name_no_ext))
            output_path = os.path.join(split_folder, video_name_no_ext)  # for split_videos per video, datasets/dataset_name/split_videos/video_name

            # SceneDetect
            if dataset == 'Epic-Kitchens':
                # EpicKitchens: shorter window because no cuts in the videos--> we need higher sensitivity to scene changes
                scene_list = detect(video_path, AdaptiveDetector(min_scene_len=60, window_width=2))
            else:
                scene_list = detect(video_path, AdaptiveDetector(min_scene_len=60, window_width=15))

            # save scenes in a JSON file
            serialized_scene_list = serialize_scenes(scene_list)
            output_json_path = os.path.join(video_folder, f"{video_name_no_ext}_scenes.json")
            with open(output_json_path, 'w') as json_file:
                json.dump(serialized_scene_list, json_file, indent=4)

            # split videos
            split_video_ffmpeg(video_path,scene_list,output_path,f'{video_name_no_ext}-Scene-$SCENE_NUMBER.mp4')


# -----------------------------------------------------------------------------------
datasets_path = ''
split_videos_path = 'split_videos'

for dataset_name in os.listdir(datasets_path):
    dataset_path = os.path.join(datasets_path, dataset_name)
    if os.path.isdir(dataset_path):
        print(f"Dataset: {dataset_name}")
        scene_detect_and_split(datasets_path, dataset_name, split_videos_path)
