import json
import csv
import os


def convert_time(time_str):
    hours, minutes, seconds = time_str.split(":")
    hours = int(hours)
    minutes = int(minutes)
    seconds = float(seconds)

    if hours > 0:
        return f"{hours} hour(s), {minutes} minute(s) and {int(seconds)} second(s)"
    elif minutes > 0:
        return f"{minutes} minute(s) and {int(seconds)} second(s)"
    else:
        return f"{int(seconds)} second(s)"


def get_sentences(json_data, version=1):
    if len(json_data) == 1:  # the json file has only one scene-->the whole video
        sentence = f"The video contains 1 scene."
    else:
        total_scenes = len(json_data)  # Get the total number of scenes
        sentence = f"The video contains {total_scenes} scenes. "

        for i, segment in enumerate(json_data):
            start_time = convert_time(segment[0])
            end_time = convert_time(segment[1])

            if version == 1:
                sentence += f"The scene {i + 1} starts at {start_time} and ends at {end_time}. "
            elif version == 2:
                sentence += f"The scene {i + 1} goes from {start_time} to {end_time}. "

    return sentence.strip()


def write_sentences(json_folder, dataset_name, output_dir, ext):
    output_file_v1 = os.path.join(output_dir, f"{dataset_name}_sentences_version1.csv")
    output_file_v2 = os.path.join(output_dir, f"{dataset_name}_sentences_version2.csv")

    with open(output_file_v1, mode="w", newline="") as file_v1, open(output_file_v2, mode="w", newline="") as file_v2:
        writer_v1 = csv.writer(file_v1)
        writer_v2 = csv.writer(file_v2)

        # Write headers for both CSV files
        writer_v1.writerow(["Video Name", "Description"])
        writer_v2.writerow(["Video Name", "Description"])

        for json_file in os.listdir(json_folder):
            if json_file.endswith(".json"):
                video_path = os.path.join(json_folder, json_file)

                # Load JSON data
                with open(video_path, "r") as f:
                    json_data = json.load(f)

                # Generate sentences for both versions
                sentence_v1 = get_sentences(json_data, version=1)
                sentence_v2 = get_sentences(json_data, version=2)

                # Get video name
                video_name = os.path.basename(json_file).split('.')[0]
                video_name2 = video_name.rsplit('_', 1)[0]  # Remove '_scene' part
                video_file = f"{video_name2}{ext}"

                # Write sentences to corresponding files
                writer_v1.writerow([video_file, sentence_v1])
                writer_v2.writerow([video_file, sentence_v2])


def get_extension(dataset_name):
    if dataset_name == 'Epic-Kitchens':
        return '.MP4'
    elif dataset_name == 'MSVD' or dataset_name == 'SumMe':
        return '.avi'
    else:
        return '.mp4'


# ------------------------------------------------------------------------------

json_scenes_path = './data/SceneDetect/scene_lists'
output_dir = './data/SceneDetect'

print('Processing:')
for dataset_folder in os.listdir(json_scenes_path):
    dataset_path = os.path.join(json_scenes_path, dataset_folder)
    if os.path.isdir(dataset_path):
        print(f"Dataset: {dataset_folder}")
        ext1 = get_extension(dataset_folder)  # get extension to later include it in the name of the videos
        write_sentences(dataset_path, dataset_folder, output_dir, ext1)
