import edge_tts
import base64
from io import BytesIO

filtered_language_dict = {
    'English': {'female': 'en-US-JennyNeural', 'male': 'en-US-GuyNeural'},
    'Chinese': {'female': 'zh-CN-XiaoxiaoNeural', 'male': 'zh-CN-YunxiNeural'},
    'French': {'female': 'fr-FR-DeniseNeural', 'male': 'fr-FR-HenriNeural'},
    'Spanish': {'female': 'es-MX-DaliaNeural', 'male': 'es-MX-JorgeNeural'},
    'Arabic': {'female': 'ar-SA-ZariyahNeural', 'male': 'ar-SA-HamedNeural'},
    'Portuguese': {'female': 'pt-BR-FranciscaNeural', 'male': 'pt-BR-AntonioNeural'},
    'Cantonese': {'female': 'zh-HK-HiuGaaiNeural', 'male': 'zh-HK-WanLungNeural'}
}

async def texttospeech(text, language, gender='female'):
    try:
        voice = filtered_language_dict[language][gender]
        communicate = edge_tts.Communicate(text=text, voice=voice, rate="+25%")
        file_path = "output.wav"
        await communicate.save(file_path)
        
        with open(file_path, "rb") as audio_file:
            audio_bytes = BytesIO(audio_file.read())
        audio = base64.b64encode(audio_bytes.read()).decode("utf-8")
        print("TTS processing completed.")
        
        audio_style = 'style="width:210px;"'
        audio_player = f'<audio src="data:audio/wav;base64,{audio}" controls autoplay {audio_style}></audio>'
        return audio_player
        
    except Exception as e:
        print(f"Error in texttospeech: {e}")
        return None