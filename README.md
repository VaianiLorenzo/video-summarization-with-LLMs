#Detecting and Mitigating Challenges in Zero-Shot Video Summarization with Video-LLMs

Video summarization aims to generate a condensed textual version of an original video.
Summaries may consist of either plain text or a shortlist of salient events, possibly including temporal or spatial references. 
Video-Large Language Models (V-LLMs) exhibit impressive zero-shot capabilities in video analysis.
However, their performance varies significantly according to the LLM prompt, the characteristics of the video, and the properties of the training data and LLM architecture. Recent works on video summarization using VL-LLMs account for plain text summaries while disregarding 
LLMs' capabilities to compose sequences of salient events and their temporal and spatial context.    
In this work, we present a new video summarization benchmark consisting of 100 videos with varying characteristics in terms of domain, duration, and spatio-temporal properties. Videos are manually annotated by three independent human experts with plain text, event-based, and spatio-temporal summaries. 
On top of the newly released dataset, we first thoroughly evaluate the zero-shot summarization performance of four state-of-the-art open-source VL-LLMs specifically designed to address spatial and temporal reasoning. Secondly, we detect and categorize the common summarization issues. Lastly, we propose different cost-effective mitigation strategies, based on Chain-of-Thought prompting, that involve the injection of knowledge extracted by external, lightweight models. The results show that VL-LLMs significantly benefit from prompting a list of recognized  actions,
whereas injecting automatically recognized objects and scene changes respectively improve spatially contextualized and event-based summaries in specific cases.

##Video-LLM

##Datasets

##Repository Structure
