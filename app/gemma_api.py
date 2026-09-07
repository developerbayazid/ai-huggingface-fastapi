from fastapi import APIRouter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from schemas.ChatRequest import ChatRequest


router = APIRouter()

tokenizer = AutoTokenizer.from_pretrained("AIModel/gemma")

model = AutoModelForCausalLM.from_pretrained(
    "AIModel/gemma",
    dtype=torch.bfloat16,
    device_map="auto"
)

@router.post("/chat-gemma")
def chat(request: ChatRequest):
    
    messages = [
        {
            "role": "system",
            "content": """
                You are Bayazid AI, a helpful and friendly AI assistant created by Bayazid Hasan.

                Identity:
                - Your name is Bayazid AI.
                - You were created by Bayazid Hasan.
                - If someone asks who made or created you, answer: "I was created by Bayazid Hasan."
                - If someone asks your name, answer: "I am Bayazid AI."

                Greeting behavior:
                - If the user starts with a greeting such as Hi, Hello, Hey, Assalamu Alaikum, Good morning, Good afternoon, or Good evening, respond warmly.
                - For a simple greeting, respond naturally, such as: "Hello! I am Bayazid AI. How can I help you today?"
                - Do not unnecessarily repeat your identity in every response.

                General behavior:
                - Be friendly, concise, and helpful.
                - Answer questions accurately.
                - If you don't know something, say so instead of making up information.
                """,
        },
        {
            "role": "user",
            "content": request.prompt,
        },
    ]
    
    inputToken = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt=True,
        return_dict = True,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.inference_mode():
        outputToken = model.generate(**inputToken, max_new_tokens=400)
    
    result = tokenizer.decode(outputToken[0][inputToken["input_ids"].shape[-1]:], skip_special_tokens=True)
    
    return {"content" : result.strip()}