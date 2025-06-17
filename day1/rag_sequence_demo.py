from transformers import AutoTokenizer, RagRetriever, RagSequenceForGeneration
import torch
from datasets import load_dataset




tokenizer = AutoTokenizer.from_pretrained("facebook/rag-sequence-nq")

inputs = tokenizer("How many people live in Paris?", return_tensors="pt")


targets = tokenizer(text_target="In Paris, there are 10 million people.", return_tensors="pt")
input_ids = inputs["input_ids"]
labels = targets["input_ids"]


retriever = RagRetriever.from_pretrained(
    "facebook/rag-sequence-nq",
    trust_remote_code=True,
    index_name="exact",
    use_dummy_dataset=True
)


# initialize with RagRetriever to do everything in one forward call
model1 = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)
outputs1 = model1(input_ids=input_ids, labels=labels)



# or use retriever separately
model2 = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", use_dummy_dataset=True)
# 1. Encode
question_hidden_states = model2.question_encoder(input_ids)[0]
# 2. Retrieve
docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
doc_scores = torch.bmm(
    question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
).squeeze(1)
# 3. Forward to generator
outputs2 = model2(
    context_input_ids=docs_dict["context_input_ids"],
    context_attention_mask=docs_dict["context_attention_mask"],
    doc_scores=doc_scores,
    decoder_input_ids=labels,
)

pass