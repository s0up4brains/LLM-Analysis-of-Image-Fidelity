import os
import base64
from anthropic import Anthropic
from dotenv import load_dotenv 
from pathlib import Path 
from tools.imagecheck import image_info
from tools.SSIM import ssim_compare
from tools.PSNR import psnr_compare
from tools.colourcomapre import delta_e_compare
from tools.contrast import contrast_compare
from tools.brightness import brightness_compare
from tools.histogram import histogram_compare

load_dotenv() #.env
client = Anthropic()    

#CALL SSIM
ssim_val = ssim_compare('data/reference.png', 'data/edited.png')
print('ssim', ssim_val)

psnr_val = psnr_compare('data/reference.png', 'data/edited.png')
print('psnr', psnr_val)

delta_e_val = delta_e_compare('data/reference.png', 'data/edited.png')
print('deltaE', delta_e_val)

contrast_val = contrast_compare('data/reference.png', 'data/edited.png')
print('contrast is', contrast_val)

brightness_val = brightness_compare('data/reference.png', 'data/edited.png')
print('brightness is', brightness_val)

histogram_val = histogram_compare('data/reference.png', 'data/edited.png')
print('histogram is', histogram_val)

reference_image_val = image_info('data/reference.png')
print('reference image', reference_image_val)

edited_image_val = image_info('data/edited.png')
print('edited image', edited_image_val)



        
# def image_block(path: str) -> dict:
#     data = base64.standard_b64encode(Path(path).read_bytes()).decode("utf-8")
#     return {
#         "type": "image",
#         "source": {"type": "base64", "media_type": "image/png", "data": data},
#     }

# message = client.messages.create(
#     model="claude-haiku-5-5",
#     max_tokens=1024,
#     messages=[
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": "Reference Image:"},
#                  image_block("reference.png"),
#                  {"type": "text", "text": "Edited Image:"},
#                  image_block("edited.png"),
#                 {
#                     "type": "text", "text": "Write a short report on how the Edited Image differs from the Reference Image (colour shift, contrast, compression artefacts, blur)",
#                     },
#             ],
#         }
#     ],
# )
# text = "".join(block.text for block in message.content if block.type == "text")
# print(text)
# print(message.stop_reason, message.usage)