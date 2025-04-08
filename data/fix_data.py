import json

# For entities/concept.json

# Read existing objects
with open("./MOOCCube/entities/concept.json", "r") as f:
    objects1 = [json.loads(line.strip()) for line in f if line.strip()]
# Write as valid JSON array
with open("./MOOCCube/entities/concept_formatted.json", "w", encoding="utf-8") as f:
    json.dump(objects1, f, indent=2, ensure_ascii=False)

# For relations/prerequisite-dependency.json
# Read existing objects
with open("./MOOCCube/relations/prerequisite-dependency.json", "r") as f:
    objects2 = [line.strip() for line in f if line.strip()]

corrected_relations = []
# Convert to correct format
for object in objects2:
    entries = object.split("\t")
    if len(entries) != 2:
        print(f"Invalid object: {object}")
        print(f"{object.split('\t')=!r}")
        continue
    entry1, entry2 = entries
    corrected_relations.append(
        {"source_concept_id": entry1, "target_concept_id": entry2}
    )

print(f"Total original objects: {len(objects2)}")
print(f"Total corrected relations: {len(corrected_relations)}")
# Write as valid JSON array
with open(
    "./MOOCCube/relations/prerequisite-dependency_formatted.json", "w", encoding="utf-8"
) as f:
    json.dump(corrected_relations, f, indent=2, ensure_ascii=False)


# For additional_information/concept_information.json
# Read existing objects
with open("./MOOCCube/additional_information/concept_infomation.json", "r") as f:
    objects3 = [json.loads(line.strip()) for line in f if line.strip()]
# Write as valid JSON array
with open(
    "./MOOCCube/additional_information/concept_infomation_formatted.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(objects3, f, indent=2, ensure_ascii=False)
