import os
import clip
import chromadb
import csv
import torch
from PIL import Image

CSV_PATH = r"c:\Users\12991\Desktop\Aplus_3\dataset1\styles.csv"
IMAGE_DIR = r"c:\Users\12991\Desktop\Aplus_3\dataset1\images"
DB_PATH = r"c:\Users\12991\Desktop\Aplus_3\vector_db"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name="products")

data_rows = []
with open(CSV_PATH, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('productDisplayName'):
            data_rows.append(row)

# idx
for idx, row in enumerate(data_rows[:10]):
    product_id = str(row["id"])
    image_path = os.path.join(IMAGE_DIR, f"{product_id}.jpg")
    text = row["productDisplayName"]

    if not os.path.exists(image_path):
        continue

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(clip.tokenize([text]).to(device))

    # l2归一化，除以norm变换之后的数，只对[1,512]最后一维干，保留维度
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    if idx == 0:
        print(f"Image vector shape: {image_features.shape}")
        print(f"Image vector first 10 values: {image_features[0][:10].cpu().numpy()}")
        print(f"Text vector shape: {text_features.shape}")
        print(f"Text vector first 10 values: {text_features[0][:10].cpu().numpy()}")

    # 归一化之后再存入chormadb
    collection.add(
        ids=[f"{product_id}_image", f"{product_id}_text"],
        embeddings=[
            image_features[0].cpu().numpy().tolist(),
            text_features[0].cpu().numpy().tolist()
        ],
        metadatas=[
            {"type": "image", "product_id": product_id},
            {"type": "text", "product_id": product_id}
        ]
    )

print(f"\nTotal items stored: {collection.count()}")