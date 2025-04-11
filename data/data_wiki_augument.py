import wikipedia
import json

with open("testing.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    try:
        wikipedia.set_lang("zh")  # Uncomment this line to set the language to Chinese
        page = wikipedia.page(item["name"])
        item["text"] = page.content
    except Exception as e:
        item["text"] = f"Error fetching article: {e}"

with open("output_testing_zh.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
