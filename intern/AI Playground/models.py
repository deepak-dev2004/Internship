from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# -------- TEXT --------
text_generator = pipeline("text-generation", model="gpt2")

def generate_text(prompt):
    return text_generator(prompt, max_length=100)[0]['generated_text']


# -------- IMAGE --------
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
image_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(image):
    inputs = processor(images=image, return_tensors="pt")
    output = image_model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)


# -------- QA --------
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

def answer_question(context, question):
    inputs = tokenizer(question, context, return_tensors="pt")
    outputs = qa_model(**inputs)

    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits) + 1

    answer = tokenizer.decode(inputs["input_ids"][0][start:end])
    return answer