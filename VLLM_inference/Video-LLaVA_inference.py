import os
import torch
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from tqdm import tqdm
import pandas as pd
import argparse

MITIGATION = "none" # "scene", "object-coco"
RANKED = True

def main():
    #videos = os.listdir("/data1/datasets/ydata-tvsum50-v1_1/video")
    root_path = "/data4/lvaiani/datasets/"
    datasets = ["TVSum", "SumMe", "MSVD", "ActivityNet", "Epic-Kitchens"]
    disable_torch_init()

    # model preparation
    model_path = 'LanguageBind/Video-LLaVA-7B'
    cache_dir = 'cache_dir'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    load_4bit, load_8bit = False, False
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
    video_processor = processor['video']
    conv_mode = "llava_v1"
    
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

            #get the scene associated with the video
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

                print("PROMPT:" + template_prompt)

                if len(template_prompt) > 5000:
                    template_prompt = template_prompt[:5000]


                inp = ' '.join([DEFAULT_IMAGE_TOKEN] * model.get_video_tower().config.num_frames) + '\n' + template_prompt

                video_path = root_path + dataset + "/selected_videos/" + video
                # video processing
                video_tensor = video_processor(video_path, return_tensors='pt')['pixel_values']
                if type(video_tensor) is list:
                    tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
                else:
                    tensor = video_tensor.to(model.device, dtype=torch.float16)

                # conversation processing
                conv = conv_templates[conv_mode].copy()
                roles = conv.roles
                conv.append_message(conv.roles[0], inp)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=tensor,
                        do_sample=True,
                        temperature=0.01,
                        max_new_tokens=1024,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria])

                while True:
                    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
                    #print(video + "-" + outputs)
                    if outputs != "</t>":
                        break

                # save outputs
                print(video + " --> " + outputs)
                print()

                generated_texts.append(outputs)
                used_datasets.append(dataset)
                used_prompt_types.append(prompt_types[i])
                used_prompts.append(template_prompt)
                used_video.append(video)

        exit()

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


if __name__ == '__main__':
    main()
