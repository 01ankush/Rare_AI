import json

# Sample input (you can also load this from a file)
with open("train.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line.strip()) for line in f.readlines()]

dataset = []

for entry in data:
    disease_name = entry.get("id", "").replace("-", " ")
    text = entry.get("text", "")
    symptoms = []
    # Extract symptoms
    for symptom_data in entry["gold"]["entities"].get("symptom_and_sign", []):
        symptom_text = symptom_data[0]
        if symptom_text not in symptoms:
            symptoms.append(symptom_text)

    if symptoms:
        dataset.append({
            "instruction" : "Below is a set of medical symptoms. Predict the most likely disease.",
            "input": ", ".join(symptoms),
            "input_label": disease_name
        })

# Output training samples
for i, sample in enumerate(dataset):
    print(f"{i+1}. Symptoms: {sample['input']}\n   Likely Disease: {sample['input_label']}\n")

# Optionally save it to a JSONL file for fine-tuning or further processing
with open("training_data_final.jsonl", "w", encoding="utf-8") as f:
    for sample in dataset:
        f.write(json.dumps(sample) + "\n")
