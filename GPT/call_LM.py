from openai import AzureOpenAI
import os
import base64
import filetype

class GPT():
    def __init__(self, type):
        if type == 'tonggpt':
            REGION = "eastus"
            API_BASE = "https://api.tonggpt.mybigai.ac.cn/proxy"
            AZURE_OPENAI_ENDPOINT = f"{API_BASE}/{REGION}"
            api_version = "2024-02-01"
            self.client = AzureOpenAI(
                                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                                api_version=api_version,
                                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                            )
    def _call_lm(self, messages, model="gpt-4o-mini-2024-07-18"): # 'gpt-4o-2024-08-06'
        return self.client.chat.completions.create(
            model=model,
            messages=messages
        ).choices[0].message.content
    
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def create_message(self, prompt, image_list=None, system_content=None):
        if system_content is None:
            content = []
            content.append({"type": "text", "text": prompt})
            if image_list is not None:
                for image in image_list:
                    base64_image = self.encode_image(image)
                    if filetype.guess(image).extension == 'png':
                        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}})
                    elif filetype.guess(image).extension == 'jpeg' or filetype.guess(image).extension == 'jpg':
                        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
            messages=[
                {"role": "user", "content": content}
            ]
        else:
            content = []
            content.append({"type": "text", "text": prompt})
            if image_list is not None:
                for image in image_list:
                    base64_image = self.encode_image(image)
                    if filetype.guess(image).extension == 'png':
                        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}})
                    elif filetype.guess(image).extension == 'jpeg' or filetype.guess(image).extension == 'jpg':
                        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": content}
            ]
        return messages

    
