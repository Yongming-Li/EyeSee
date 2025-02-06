import json
import requests
import base64


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def get_gpt_response(api_key, image_path, prompt, history=None):

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    if history:
        if len(history) > 4:
            history = history[-4:]
    else:
        history = []
    
    messages = history[:]
    base64_images = []
    
    if image_path:
        if isinstance(image_path, list):
            for img in image_path:
                base64_image = encode_image(img)
                base64_images.append(base64_image)
        else:
            base64_image = encode_image(image_path)
            base64_images.append(base64_image)
        
        messages.append({
            "role": "user",
            "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_images}"
                        }
                    }
                ]
        })
    else: 
        messages.append({
            "role": "user",
            "content": prompt
        })
                
    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 600
    }
    

    # Sending the request to the OpenAI API
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    result = response.json()
    print("gpt result",result)
    try:
        content = result['choices'][0]['message']['content']
        if content.startswith("```json"):
                content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        return content
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        return json.dumps({"error": "Failed to parse model output", "details": str(e)})
