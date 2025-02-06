import re
import emoji
from .utils import get_gpt_response

async def get_artistinfo(artist_name, api_key, state, language, autoplay, length, log_state, texttospeech_fn):
    prompt = (
        f"Provide a concise summary of about {length} words in {language} on the painter {artist_name}, "
        "covering his biography, major works, artistic style, significant contributions to the art world, "
        "and any major awards or recognitions he has received. Start your response with 'Artist Background: '."
    )
    
    res = get_gpt_response(api_key, None, prompt)
    state = state + [(None, res)]
    read_info = re.sub(r'[#[\]!*]', '', res)
    read_info = emoji.replace_emoji(read_info, replace="")    
    log_state = log_state + [(f"res", None)]

    if autoplay:
        audio_output = await texttospeech_fn(read_info, language)
        return state, state, audio_output, log_state
    return state, state, None, log_state

async def get_yearinfo(year, api_key, state, language, autoplay, length, log_state, texttospeech_fn):
    prompt = (
        f"Provide a concise summary of about {length} words in {language} on the art historical period "
        f"associated with the year {year}, covering its major characteristics, influential artists, "
        "notable works, and its significance in the broader context of art history with 'History Background: '."
    )
    
    res = get_gpt_response(api_key, None, prompt)
    log_state = log_state + [(f"res", None)]
    state = state + [(None, res)]
    read_info = re.sub(r'[#[\]!*]', '', res)
    read_info = emoji.replace_emoji(read_info, replace="")    

    if autoplay:
        audio_output = await texttospeech_fn(read_info, language)
        return state, state, audio_output, log_state
    return state, state, None, log_state