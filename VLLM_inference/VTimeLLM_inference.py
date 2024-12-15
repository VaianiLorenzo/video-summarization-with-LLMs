import os
import sys
import argparse
import torch
from vtimellm.constants import IMAGE_TOKEN_INDEX
from vtimellm.conversation import conv_templates, SeparatorStyle
from vtimellm.model.builder import load_pretrained_model, load_lora
from vtimellm.utils import disable_torch_init
from vtimellm.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, VideoExtractor
from PIL import Image
import requests
from io import BytesIO
from transformers import TextStreamer
from easydict import EasyDict as edict
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
import numpy as np
import clip
import pandas as pd
from tqdm import tqdm

MITIGATION = "action" # "scene", "object-coco"
RANKED = False

def inference(model, image, query, tokenizer):
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image[None,].cuda(),
            do_sample=True,
            temperature=0.05,
            num_beams=1,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            use_cache=True)

        # https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1295

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs



def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--clip_path", type=str, default="checkpoints/clip/ViT-L-14.pt")
    parser.add_argument("--model_base", type=str, default="/path/to/vicuna-7b-v1.5")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default="checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--stage2", type=str, default="checkpoints/vtimellm-vicuna-v1-5-7b-stage2")
    parser.add_argument("--stage3", type=str, default="checkpoints/vtimellm-vicuna-v1-5-7b-stage3")
    parser.add_argument("--video_path", type=str, default="images/demo.mp4")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    disable_torch_init()
    tokenizer, model, context_len = load_pretrained_model(args, args.stage2, args.stage3)
    model = model.cuda()
    # model.get_model().mm_projector.to(torch.float16)
    model.to(torch.float16)

    clip_model, _ = clip.load(args.clip_path)
    clip_model.eval()
    clip_model = clip_model.cuda()

    video_loader = VideoExtractor(N=100)

    root_path = "/data4/lvaiani/datasets/"
    datasets = ["TVSum", "SumMe", "MSVD", "ActivityNet", "Epic-Kitchens"]
    datasets = ["Epic-Kitchens"]


    generated_texts = []
    used_prompt_types = []
    used_prompts = []
    used_datasets = []
    used_video = []

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
        videos = [video.replace(" .mp4", ".mp4") for video in videos if not video.startswith('.')]

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
            print("Processing video: " + video)

            if "scene" in MITIGATION:
                scene_description = scene_df[scene_df['Video Name'] == video]['Description'].values[0]
            if "object-coco" in MITIGATION:
                object_description = object_df[object_df['Video name'] == video]['Description'].values[0]
            if "action" in MITIGATION:
                action_description = action_df[action_df['video'] == video.split(".")[0]]['description'].values[0]

            for i in tqdm(range(len(prompts_list))):
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

                _, images = video_loader.extract({'id': None, 'video': video_path})

                transform = Compose([
                    Resize(224, interpolation=BICUBIC),
                    CenterCrop(224),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ])

                # print(images.shape) # <N, 3, H, W>
                images = transform(images / 255.0)
                images = images.to(torch.float16)
                with torch.no_grad():
                    features = clip_model.encode_image(images.to('cuda'))

                output = inference(model, features, "<video>\n " + template_prompt, tokenizer)

                print(video + " --> " + output)
                generated_texts.append(output)
                used_datasets.append(dataset)
                used_prompt_types.append(prompt_types[i])
                used_prompts.append(template_prompt)
                used_video.append(video)

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


####
# conda activate vtimellm
# python -m vtimellm.AMAZON_inference --model_base /home/lvaiani/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d
####