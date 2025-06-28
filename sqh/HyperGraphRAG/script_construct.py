import os
import json
from hypergraphrag import HyperGraphRAG
#os.environ["OPENAI_API_KEY"] = "sk-vAURH343sL2qrvWK2aAd5e038dE14e158b54B9E6E6352086"
#os.environ["OPENAI_API_BASE"] = "https://aihubmix.com/v1"
rag = HyperGraphRAG(working_dir=f"expr/example_sql")

with open(f"example_contexts.json", mode="r", encoding="utf-8") as f:
    unique_contexts = json.load(f)
    
rag.insert(unique_contexts)