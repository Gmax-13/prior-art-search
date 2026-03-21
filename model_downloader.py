import os
os.environ["HF_HOME"] = "D:/huggingface_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from sentence_transformers import SentenceTransformer, CrossEncoder

print("Downloading SPECTER (bi-encoder)...")
SentenceTransformer("sentence-transformers/allenai-specter")
print("SPECTER done.\n")

print("Downloading CrossEncoder...")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
print("CrossEncoder done.\n")

print("All models cached to D:/huggingface_cache")