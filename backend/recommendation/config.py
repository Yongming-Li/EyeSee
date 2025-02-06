import torch
from transformers import AutoProcessor, SiglipModel
from huggingface_hub import hf_hub_download
import faiss
import pandas as pd

class RecommendationConfig:
    def __init__(self):
        hf_hub_download("merve/siglip-faiss-wikiart", "siglip_10k_latest.index", local_dir="./")
        hf_hub_download("merve/siglip-faiss-wikiart", "wikiart_10k_latest.csv", local_dir="./")

        self.index = faiss.read_index("./siglip_10k_latest.index")
        self.df = pd.read_csv("./wikiart_10k_latest.csv")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
        self.model = SiglipModel.from_pretrained("google/siglip-base-patch16-224").to(self.device)

    def get_messages(self, language):
        return {
            "English": "🖼️ Please refer to the section below to see the recommended results.",
            "Chinese": "🖼️  请到下方查看推荐结果。"
        }[language]