from sentence_transformers import SentenceTransformer

# Load https://huggingface.co/sentence-transformers/all-mpnet-base-v2
model = SentenceTransformer("LaBSE")

embeddings = model.encode([
    "Nuclear fission is a reaction in which the nucleus of an atom splits into two or more smaller nuclei.",
    "அணுக்கரு பிளவு என்பது ஒரு அணுவின் கரு இரண்டு அல்லது அதற்கு மேற்பட்ட சிறிய கருக்களாகப் பிளவுபடும் ஒரு வினை ஆகும்.",
    "核分裂とは、原子核が2つ以上のより小さな原子核に分裂する反応のことである。", # Japanese
    "核裂变是一种反应，其中一个原子核分裂成两个或多个较小的原子核。" # Chinese (Simplified)
])

similarities = model.similarity([embeddings[0]], embeddings)

print(similarities)

