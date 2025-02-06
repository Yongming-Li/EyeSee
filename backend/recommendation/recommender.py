import torch
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import spaces
import gradio as gr
import re
import emoji
from ..prompts.prompt_templates import PromptTemplates
import faiss

class ImageRecommender:
    def __init__(self, config):
        self.config = config
        
    def read_image_from_url(self, url):
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
        
    def extract_features_siglip(self, image):
        with torch.no_grad():
            inputs = self.config.processor(images=image, return_tensors="pt").to(self.config.device)
            image_features = self.config.model.get_image_features(**inputs)
        return image_features
        
    def process_image(self, image_path, num_results=2):
        input_image = Image.open(image_path).convert("RGB")
        input_features = self.extract_features_siglip(input_image)
        input_features = input_features.detach().cpu().numpy()
        input_features = np.float32(input_features)
        faiss.normalize_L2(input_features)
        
        distances, indices = self.config.index.search(input_features, num_results)
        gallery_output = []
        
        for i, v in enumerate(indices[0]):
            sim = -distances[0][i]
            image_url = self.config.df.iloc[v]["Link"]
            img_retrieved = self.read_image_from_url(image_url)
            gallery_output.append(img_retrieved)
            
        return gallery_output
        
    @spaces.GPU
    def infer(self, crop_image_path, full_image_path, state, language, task_type=None):
        style_gallery_output = []
        item_gallery_output = []
        
        if crop_image_path:
            item_gallery_output = self.process_image(crop_image_path, 2)
            style_gallery_output = self.process_image(full_image_path, 2)
        else:
            style_gallery_output = self.process_image(full_image_path, 4)
            
        msg = self.config.get_messages(language)
        state += [(None, msg)]
        
        return item_gallery_output, style_gallery_output, state, state

    async def item_associate(self, new_crop, openai_api_key, language, autoplay, length, 
                           log_state, sort_score, narrative, state, evt: gr.SelectData):
        rec_path = evt._data['value']['image']['path']
        return (
            state,
            state,
            None,
            log_state,
            None,
            gr.update(value=[]),
            rec_path,
            rec_path,
            "Item"
        )

    async def style_associate(self, image_path, openai_api_key, language, autoplay, 
                            length, log_state, sort_score, narrative, state, artist, 
                            evt: gr.SelectData):
        rec_path = evt._data['value']['image']['path']
        return (
            state,
            state,
            None,
            log_state,
            None,
            gr.update(value=[]),
            rec_path,
            rec_path,
            "Style"
        )

    def generate_recommendation_prompt(self, recommend_type, narrative, language, length, artist=None):

        narrative_value = PromptTemplates.NARRATIVE_MAPPING[narrative]
        prompt_type = 0 if recommend_type == "Item" else 1
        
        if narrative_value == 1 and recommend_type == "Style":
            return PromptTemplates.RECOMMENDATION_PROMPTS[prompt_type][narrative_value].format(
                language=language,
                length=length,
                artist=artist[8:] if artist else ""
            )
        else:
            return PromptTemplates.RECOMMENDATION_PROMPTS[prompt_type][narrative_value].format(
                language=language,
                length=length
            )
