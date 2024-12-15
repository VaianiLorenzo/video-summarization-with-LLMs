"""
How to run this file:

cd VideoChatGPT
python -m video_chatgpt.single_video_inference \
    --model-name <path of llava weights, for eg "LLaVA-7B-Lightening-v1-1"> \
    --projection_path <path of projection for eg "video-chatgpt-weights/video_chatgpt-7B.bin"> \
    --video_path <video_path>
"""

from video_chatgpt.video_conversation import conv_templates, SeparatorStyle
from video_chatgpt.model.utils import KeywordsStoppingCriteria
import torch

#add new packages as below
from PIL import Image
from decord import VideoReader, cpu
from video_chatgpt.eval.model_utils import initialize_model, load_video
import argparse
import numpy as np
import os
from tqdm import tqdm
import pandas as pd


MITIGATION = "action" # "scene", "object-coco"
RANKED = False

# Define constants
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"



def get_spatio_temporal_features_torch(features):
    """
    Computes spatio-temporal features from given features.

    Parameters:
    features (torch.Tensor): Input features to process.

    Returns:
    torch.Tensor: Spatio-temporal features.
    """

    # Extract the dimensions of the features
    t, s, c = features.shape

    # Compute temporal tokens as the mean along the time axis
    temporal_tokens = torch.mean(features, dim=1)

    # Padding size calculation
    padding_size = 100 - t

    # Pad temporal tokens if necessary
    if padding_size > 0:
        padding = torch.zeros(padding_size, c, device=features.device)
        temporal_tokens = torch.cat((temporal_tokens, padding), dim=0)

    # Compute spatial tokens as the mean along the spatial axis
    spatial_tokens = torch.mean(features, dim=0)

    # Concatenate temporal and spatial tokens and cast to half precision
    concat_tokens = torch.cat([temporal_tokens, spatial_tokens], dim=0).half()

    return concat_tokens


def video_chatgpt_infer(video_frames, question, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len):
    """
    Run inference using the Video-ChatGPT model.

    Parameters:
    sample : Initial sample
    video_frames (torch.Tensor): Video frames to process.
    question (str): The question string.
    conv_mode: Conversation mode.
    model: The pretrained Video-ChatGPT model.
    vision_tower: Vision model to extract video features.
    tokenizer: Tokenizer for the model.
    image_processor: Image processor to preprocess video frames.
    video_token_len (int): The length of video tokens.

    Returns:
    dict: Dictionary containing the model's output.
    """

    # Prepare question string for the model
    if model.get_model().vision_config.use_vid_start_end:
        qs = question + '\n' + DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len + DEFAULT_VID_END_TOKEN
    else:
        qs = question + '\n' + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len

    # Prepare conversation prompt
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize the prompt
    inputs = tokenizer([prompt])

    # Preprocess video frames and get image tensor
    image_tensor = image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values']

    # Move image tensor to GPU and reduce precision to half
    image_tensor = image_tensor.half().cuda()

    # Generate video spatio-temporal features
    with torch.no_grad():
        image_forward_outs = vision_tower(image_tensor, output_hidden_states=True)
        frame_features = image_forward_outs.hidden_states[-2][:, 1:] # Use second to last layer as in LLaVA
    video_spatio_temporal_features = get_spatio_temporal_features_torch(frame_features)

    # Move inputs to GPU
    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    # Define stopping criteria for generation
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    # Run model inference
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            video_spatio_temporal_features=video_spatio_temporal_features.unsqueeze(0),
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            stopping_criteria=[stopping_criteria])

    # Check if output is the same as input
    n_diff_input_output = (input_ids != output_ids[:, :input_ids.shape[1]]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

    # Decode output tokens
    outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

    # Clean output string
    outputs = outputs.strip().rstrip(stop_str).strip()

    return outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--vision_tower_name", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--projection_path", type=str, required=False, default="")
    parser.add_argument("--conv_mode", type=str, required=False, default='video-chatgpt_v1')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    model, vision_tower, tokenizer, image_processor, video_token_len = \
        initialize_model(args.model_name, args.projection_path)

    root_path = "/data4/lvaiani/datasets/"
    datasets = ["TVSum", "SumMe", "MSVD", "ActivityNet", "Epic-Kitchens"]
    datasets = ["Epic-Kitchens"]

    generated_texts = []
    used_prompt_types = []
    used_prompts = []
    used_datasets = []
    used_video = []
    counter = 0
    # prompt_types = ["Plain-text summary", "Timeline summary", "Spatially contextualized summary", "Spatio-Temporal contextualized summary", "Saliency", "Conciseness", "Temporal order", "Temporal order + latency", "Saliency + Conciseness", "Saliency + Conciseness + Order", "Saliency + Conciseness + Order + Latency", "Structured timeline summary", "Structured spatially contextualized summary", "Structured spatio-temporal contextualized summary", "Spatio-temporal contextualized summary 1", "structered spatio-temporal contextualized summary 2"]
    # prompts_list = ["Summarize the video.", 
    #                 "Provide the timeline of salient events or actions occurring in the video.", 
    #                 "Provide a description of the video grouping together the events occurring in the place/location.", 
    #                 "Provide a summary highlighting the main spatio-temporal contexts in the video.", 
    #                 "Enumerate the salient events or actions occurring in the video.", 
    #                 "Identify the repeated events or actions in the video.", 
    #                 "Provide a sequence of salient events or actions happening in the video.", 
    #                 "Provide a sequence of salient events or actions in the video indicating the exact time range in which each event or action happens.", 
    #                 "Highlight the most important events or actions in the video, ensuring that no redundant information is included.", 
    #                 "Enumerate the key moments in the video in the order they occur, ensuring that the list is concise and free of redundant details.", 
    #                 "Provide a timestamped, concise list of the key events or actions in the video, ensuring the sequence is in chronological order and free of redundant information.",
    #                 "Provide the timeline of salient events or actions occurring in the video. Use the format: <start-timestamp>-<end-timestamp>: <event/action>.",
    #                 "Provide a description of the video grouping together the events occurring in the place/location. Use the format: <place/location>: <event/action>.",
    #                 "Provide a summary highlighting the main spatio-temporal contexts in the video. Use the format: <place/location>-<start-timestamp>-<end-timestamp>: <event/action>.", 
    #                 "Provide a summary highlighting the main spatio-temporal contexts in the video. For every event that occours in the video provide details about the location, start time, end time, and the event/action.",
    #                 "Provide a summary highlighting the main spatio-temporal contexts in the video. For every event that occours in the video provide details about the location, start time, end time, and the event/action. Use the format: <place/location>-<start-timestamp>-<end-timestamp>: <event/action>."]

    prompt_types = ["Plain-text summary", "Saliency", "Spatially contextualized summary", "Structured timeline summary", "structered spatio-temporal contextualized summary 2"]
    prompts_list = ["Summarize the video.",  
                    "Enumerate the salient events or actions occurring in the video.", 
                    "Provide a description of the video grouping together the events occurring in the place/location.",
                    "Provide the timeline of salient events or actions occurring in the video. Use the format: <start-timestamp>-<end-timestamp>: <event/action>.",
                    "Provide a summary highlighting the main spatio-temporal contexts in the video. For every event that occours in the video provide details about the location, start time, end time, and the event/action. Use the format: <place/location>-<start-timestamp>-<end-timestamp>: <event/action>."]
                    
    for dataset in datasets:
        print("Processing dataset: " + dataset)
        videos = os.listdir(root_path + dataset + "/selected_videos")
        videos = [video for video in videos if not video.startswith('.')]

        if "scene" in MITIGATION:
            scene_df = pd.read_csv("/home/lvaiani/amazon/scene_detection/" + dataset + "_sentences_version_1.csv")
        if "object-coco" in MITIGATION:
            if RANKED:
                object_df = pd.read_csv("/home/lvaiani/amazon/object_detection/" + dataset + "_sentences_coco_ranked.csv")
            else:
                object_df = pd.read_csv("/home/lvaiani/amazon/object_detection/" + dataset + "_sentences_coco.csv")
        if "action" in MITIGATION:
            if dataset == "Epic-Kitchens":
                action_df = pd.read_csv("/home/lvaiani/amazon/action_detection/" + dataset + "_action_sentences_tada_v2.csv")
            else:
                action_df = pd.read_csv("/home/lvaiani/amazon/action_detection/" + dataset + "_action_sentences_mmaction2.csv")

        for video in videos:
            print("\tProcessing video: " + video)

            if "scene" in MITIGATION:
                scene_description = scene_df[scene_df['Video Name'] == video]['Description'].values[0]
            if "object-coco" in MITIGATION:
                object_description = object_df[object_df['Video name'] == video]['Description'].values[0]
            if "action" in MITIGATION:
                action_description = action_df[action_df['video'] == video.split(".")[0]]['description'].values[0]
                
            for i in range(len(prompts_list)):
                template_prompt = prompts_list[i]
                template_prompt = template_prompt + " " + scene_description if "scene" in MITIGATION else template_prompt
                template_prompt = template_prompt + " The following actions have been detected in the video: " + action_description if "action" in MITIGATION else template_prompt
                if RANKED:
                    template_prompt = template_prompt + " Some details about the elements in this video: " + object_description if "object-coco" in MITIGATION else template_prompt
                else:
                    template_prompt = template_prompt + " Some details about the elements in this video: " + ":".join(object_description.split(":")[1:]) if "object-coco" in MITIGATION else template_prompt
                
                print("Prompt length: ", len(template_prompt))

                if len(template_prompt) > 10000:
                    template_prompt = template_prompt[:10000]

                video_path = root_path + dataset + "/selected_videos/" + video
                video_frames = load_video(video_path)
                conv_mode = args.conv_mode
                try:
                    # Run inference on the video and add the output to the list
                    output = video_chatgpt_infer(video_frames, template_prompt, conv_mode, model, vision_tower,
                                                        tokenizer, image_processor, video_token_len)
                    print(video + " --> " + output)
                    generated_texts.append(output)
                    used_datasets.append(dataset)
                    used_prompt_types.append(prompt_types[i])
                    used_prompts.append(template_prompt)
                    used_video.append(video)
                    
                except Exception as e:
                    print(f"Error processing video file '{video_path}': {e}")

        df = pd.DataFrame({'dataset': used_datasets, 'video': used_video, 'prompt_type': used_prompt_types, 'prompt': used_prompts, 'summary': generated_texts})
        output_name = 'AMAZON_inference2'
        if "scene" in MITIGATION:
            output_name = output_name + "_mit-scene"
        if "object-coco" in MITIGATION:
            output_name = output_name + "_mit-object-coco"
        if RANKED:
            output_name = output_name + "_ranked"
        if "action" in MITIGATION:
            output_name = output_name + "_mit-action"
        output_name = output_name + ".tsv"
        df.to_csv(output_name, index=False, sep='\t')        
        counter += 1


# RUN
# pip install -r requirements.txt 
# export PYTHONPATH="./:$PYTHONPATH"
# CUDA_VISIBLE_DEVICES=1 python video_chatgpt/AMAZON_inference.py --model-name LLaVA-7B-Lightening-v1-1