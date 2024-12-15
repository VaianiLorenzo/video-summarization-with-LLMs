# Detecting and Mitigating Challenges in Zero-Shot Video Summarization with Video LLMs

Video summarization aims to generate a condensed textual version of an original video.
Summaries may consist of either plain text or a shortlist of salient events, possibly including temporal or spatial references. 
Video-Large Language Models (VLLMs) exhibit impressive zero-shot capabilities in video analysis.
However, their performance varies significantly according to the LLM prompt, the characteristics of the video, and the properties of the training data and LLM architecture. Recent works on video summarization using VLLMs account for plain text summaries while disregarding 
LLMs' capabilities to compose sequences of salient events and their temporal and spatial context.    
In this work, we present a new video summarization benchmark consisting of 100 videos with varying characteristics in terms of domain, duration, and spatio-temporal properties. Videos are manually annotated by three independent human experts with plain text, event-based, and spatio-temporal summaries. 
On top of the newly released dataset, we first thoroughly evaluate the zero-shot summarization performance of four state-of-the-art open-source VLLMs specifically designed to address spatial and temporal reasoning. Secondly, we detect and categorize the common summarization issues. Lastly, we propose different cost-effective mitigation strategies, based on Chain-of-Thought prompting, that involve the injection of knowledge extracted by external, lightweight models. The results show that VLLMs significantly benefit from prompting a list of recognized  actions,
whereas injecting automatically recognized objects and scene changes respectively improve spatially contextualized and event-based summaries in specific cases.

## Video LLMs

In this work, we explored the performance of 4 different V-LLM.

| Model         | Repository Link                            | Paper Link                                   |
|------------------|-------------------------------------------|---------------------------------------------|
| Video-ChatGPT            | [GitHub](https://github.com/mbzuai-oryx/Video-ChatGPT) | [Paper](https://arxiv.org/abs/2306.05424)   |
| VideoLLaVA           | [GitHub](https://github.com/PKU-YuanGroup/Video-LLaVA)     | [Paper](https://arxiv.org/abs/2311.10122)   |
| Video-LLaMA2 | [GitHub](https://github.com/DAMO-NLP-SG/VideoLLaMA2) | [Paper](https://arxiv.org/abs/2406.07476)   |
| VTimeLLM | [GitHub](https://github.com/huangb23/VTimeLLM) | [Paper](https://arxiv.org/abs/2311.18445) |

## Datasets
The videos involved in our V-LLMs are sampled from the following datasets:
- TVSum
- SumMe
- ActivityNet
- MSVD
- Epic-Kitchens

## Repository Structure

This repository contains the following 3 main folders:

#### Directory `LLM_inference`
This folder contain the scripts used to run each VLLM inference. Each script must replace the main inference script in the official repository of the corresponding VLLM. All teh dependencies are the same as the original VLLM model.  

#### Directory `mitigation_techniques`
This folder contains the scripts to extract useful information and to create the sentences that can be injected in the VLLM prompt to mitigate the final summary generation. In particular:
- Directory `scene`: contains the script used to divide each video in separate scene, saving corresponding start and end timestamps. 
- Directory `object`: contains the script to extract relevan object from the video usig the YOLO architecture.
- Directory `scene`: contains the scripts to detect relevant actions witih the original video, using the mmmaction2 model.

#### Directory `annotations`
Contains the new event-based annotations used to evaluate V-LLMs. Our annotation will be released upon paper acceptance. 

