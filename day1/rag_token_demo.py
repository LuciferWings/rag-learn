from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
import torch

tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq")

retriever = RagRetriever.from_pretrained(
    "facebook/rag-token-nq",
    trust_remote_code=True,
    index_name="exact",
    use_dummy_dataset=True
)

model.set_retriever(retriever)

input_text = "What is the capitial of France?"
inputs = tokenizer(input_text, return_tensors="pt")

with torch.no_grad():
    generated_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        num_return_sequences=1,
        max_length=50
    )

answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
