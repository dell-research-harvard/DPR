from transformers import pipeline

model = "facebook/bart-large"
classifier = pipeline("zero-shot-classification", model=model, device=0)

candidate_labels = ["Vietnam war", "Other topic"]

classifier("I like pasta", candidate_labels)