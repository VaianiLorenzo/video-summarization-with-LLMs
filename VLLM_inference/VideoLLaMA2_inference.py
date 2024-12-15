import torch
import os
import sys
sys.path.append('./')
from videollama2.conversation import conv_templates, SeparatorStyle
from videollama2.constants import DEFAULT_MMODAL_TOKEN, MMODAL_TOKEN_INDEX
from videollama2.mm_utils import get_model_name_from_path, tokenizer_MMODAL_token, KeywordsStoppingCriteria, process_video, process_image
from videollama2.model.builder import load_pretrained_model
import pandas as pd
from tqdm import tqdm
from transformers import logging
logging.set_verbosity_error()

MITIGATION = "action" # "scene", "object-coco"
RANKED = False

def model_init():
        # 1. Initialize the model.
        model_path = 'DAMO-NLP-SG/VideoLLaMA2-7B-Base'
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name)
        model = model.to('cuda')
        conv_mode = 'llama_2'
        return model, tokenizer, processor, conv_mode


def inference(model, tokenizer, processor, conv_mode, prompt, video_path):
    
    # Video Inference
    paths = [video_path]
    questions = [prompt]
    modal_list = ['video']

    # 2. Visual preprocess (load & transform image or video).
    tensor = process_video(paths[0], processor, model.config.image_aspect_ratio).to(dtype=torch.float16, device='cuda', non_blocking=True)
    default_mm_token = DEFAULT_MMODAL_TOKEN["VIDEO"]
    modal_token_index = MMODAL_TOKEN_INDEX["VIDEO"]
    tensor = [tensor]

    # 3. Text preprocess (tag process & generate prompt).
    question = default_mm_token + "\n" + questions[0]
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_MMODAL_token(prompt, tokenizer, modal_token_index, return_tensors='pt').unsqueeze(0).to('cuda:0')

    # 4. Generate a response according to visual signals and prompts. 
    stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE] else conv.sep2
    # keywords = ["<s>", "</s>"]
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)


    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images_or_videos=tensor,
            modal_list=modal_list,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return outputs[0]


if __name__ == "__main__":
    model, tokenizer, processor, conv_mode = model_init()
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
                output = inference(model, tokenizer, processor, conv_mode, template_prompt, video_path)
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

# RUN
# pip install -e .
# pip install -r requirements.txt
# pip install torch --upgrade
