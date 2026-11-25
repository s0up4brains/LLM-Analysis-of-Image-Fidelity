import os
import base64
from anthropic import Anthropic
from dotenv import load_dotenv 
from pathlib import Path 

load_dotenv() #.env
client = Anthropic()    
        
def image_block(path: str) -> dict:
    data = base64.standard_b64encode(Path(path).read_bytes()).decode("utf-8")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": data},
    }

message = client.messages.create(
    model="claude-haiku-5-5",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Reference Image:"},
                 image_block("reference.png"),
                 {"type": "text", "text": "Edited Image:"},
                 image_block("edited.png"),
                {
                    "type": "text", "text": "Write a short report on how the Edited Image differs from the Reference Image (colour shift, contrast, compression artefacts, blur)",
                    },
            ],
        }
    ],
)
text = "".join(block.text for block in message.content if block.type == "text")
print(text)
print(message.stop_reason, message.usage)