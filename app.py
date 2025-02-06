import datetime
from io import BytesIO
import io
from math import inf
import os
import base64
import json
import gradio as gr
import numpy as np
from PIL import Image
import emoji
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from caption_anything.model import CaptionAnything
from caption_anything.utils.utils import mask_painter, seg_model_map, prepare_segmenter, image_resize
from caption_anything.utils.parser import parse_augment
from caption_anything.captioner import build_captioner
from caption_anything.segmenter import build_segmenter
from backend.chatbox import build_chatbot_tools, get_new_image_name
from segment_anything import sam_model_registry
import easyocr
import re
from langchain import __version__
import pandas as pd
import os
import json
import numpy as np
from PIL import Image
from backend.prompts import generate_prompt
from backend.recommendation import RecommendationConfig, ImageRecommender
from backend.gpt_service import get_gpt_response, get_artistinfo, get_yearinfo
from backend.texttospeech.tts import texttospeech
from backend.prompts.prompt_templates import PromptTemplates

recommendation_config = RecommendationConfig()
recommender = ImageRecommender(recommendation_config)

MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "4096"))

args = parse_augment()
args.segmenter = "huge"
args.segmenter_checkpoint = "sam_vit_h_4b8939.pth"
args.clip_filter = True

try:
    print("Before preparing segmenter")
    if args.segmenter_checkpoint is None:
        _, segmenter_checkpoint = prepare_segmenter(args.segmenter)
    else:
        segmenter_checkpoint = args.segmenter_checkpoint
    print("After preparing segmenter")
except Exception as e:
    print(f"Error in preparing segmenter: {e}")

try:
    print("Before building captioner")
    shared_captioner = build_captioner(args.captioner, args.device, args)
    print("After building captioner")
except Exception as e:
    print(f"Error in building captioner: {e}")

try:
    print("Before loading SAM model")
    shared_sam_model = sam_model_registry[seg_model_map[args.segmenter]](checkpoint=segmenter_checkpoint).to(args.device)
    print("After loading SAM model")
except Exception as e:
    print(f"Error in loading SAM model: {e}")

try:
    print("Before initializing OCR reader")
    ocr_lang = ["ch_tra", "en"]
    shared_ocr_reader = easyocr.Reader(ocr_lang,model_storage_directory=".EasyOCR/model")
    print("After initializing OCR reader")
except Exception as e:
    print(f"Error in initializing OCR reader: {e}")

try:
    print("Before building chatbot tools")
    tools_dict = {e.split('_')[0].strip(): e.split('_')[1].strip() for e in args.chat_tools_dict.split(',')}
    shared_chatbot_tools = build_chatbot_tools(tools_dict)
    print("After building chatbot tools")
except Exception as e:
    print(f"Error in building chatbot tools: {e}")

query_focus_en = [
    "Provide a description of the item.",
    "Provide a description and analysis of the item.",
    "Provide a description, analysis, and interpretation of the item.",
    "Evaluate the item."
]

query_focus_zh = [
    "请描述一下这个物体。",
    "请描述和分析一下这个物体。",
    "请描述、分析和解释一下这个物体。",
    "请以艺术鉴赏的角度评价一下这个物体。"
]

def build_caption_anything_with_models(args, api_key="", captioner=None, sam_model=None, ocr_reader=None, text_refiner=None,
                                       session_id=None):
    segmenter = build_segmenter(args.segmenter, args.device, args, model=sam_model)
    captioner = captioner
    if session_id is not None:
        print('Init caption anything for session {}'.format(session_id))
    return CaptionAnything(args, api_key, captioner=captioner, segmenter=segmenter, ocr_reader=ocr_reader, text_refiner=text_refiner)


def validate_api_key(api_key):
    api_key = str(api_key).strip()
    print(api_key)
    try:
        test_llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=api_key)
        print("test_llm")
        response = test_llm([HumanMessage(content='Hello')])
        print(response)
        return True
    except Exception as e:
        print(f"API key validation failed: {e}")
        return False


async def init_openai_api_key(api_key=""):
    if api_key and len(api_key) > 30:
        print(api_key)
        if validate_api_key(api_key):
            try:
                # text_refiner = build_text_refiner(args.text_refiner, args.device, args, api_key)
                text_refiner = None
                print("text refiner")
            except Exception as e:
                print(f"Error initializing TextRefiner or ConversationBot: {e}")
                text_refiner = None
            return [gr.update(visible=True)]+[gr.update(visible=False)]+[gr.update(visible=True)]* 3 + [gr.update(visible=False)]*2 + [text_refiner, None]+[gr.update(visible=True)]*4+[gr.update(visible=False)]+[gr.update(visible=True)]*2
        else:
            print("Invalid API key.")
    else:
        print("API key is too short.")
        return [gr.update(visible=False)]*5 + [gr.update(visible=True)]*2 + [text_refiner, 'Your OpenAI API Key is not available']+[gr.update(visible=False)]*7
       
        
def get_click_prompt(chat_input, click_state, click_mode):
    inputs = json.loads(chat_input)
    if click_mode == 'Continuous':
        points = click_state[0]
        labels = click_state[1]
        for input in inputs:
            points.append(input[:2])
            labels.append(input[2])
    elif click_mode == 'Single':
        points = []
        labels = []
        for input in inputs:
            points.append(input[:2])
            labels.append(input[2])
        click_state[0] = points
        click_state[1] = labels
    else:
        raise NotImplementedError

    prompt = {
        "prompt_type": ["click"],
        "input_point": click_state[0],
        "input_label": click_state[1],
        "multimask_output": "True",
    }
    return prompt


def update_click_state(click_state, caption, click_mode):
    if click_mode == 'Continuous':
        click_state[2].append(caption)
    elif click_mode == 'Single':
        click_state[2] = [caption]
    else:
        raise NotImplementedError

async def chat_input_callback(*args):
    chat_input, state, aux_state ,language , autoplay,gender,api_key,image_input,log_state,history,persona = args
    message = chat_input["text"]
    if persona == "Narrator":
        prompt="Please help me answer the question with this painting {question} in {language}, with a response length of about 70 words.."
    elif persona =="Artist":
        prompt="When generating the answer, you should tell others that you are one of the creators of these paintings and generate the text in the tone and manner as if you are the creator of the painting.  Please help me answer the question with this painting {question} in {language}, with a response length of about 70 words."
    else:
        prompt="When generating answers, you should tell people that you are the object itself that was selected, and generate text in the tone and manner in which you are the object or the person. Please help me answer the question with this painting {question} in {language}, with a response length of about 70 words."
    prompt=prompt.format(question=message, language=language)
    
    result=get_gpt_response(api_key, image_input,prompt+message,history)
    read_info = re.sub(r'[#[\]!*]','',result)
    read_info = emoji.replace_emoji(read_info,replace="")   
    state = state + [(message,result)]
    log_state += [(message,"/////")]
    log_state += [("/////",result)]
    
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": result})
    
    if autoplay==False:
        return state, state, aux_state, None,log_state,history
        
    else: 
        audio = await texttospeech(read_info,language,gender)
        return state, state, aux_state, audio,log_state,history

        
async def upload_callback(image_input,state, log_state, task_type, openai_api_key=None,language="English",narritive=None,history=None,autoplay=True,session="Session 1"):
    print("narritive", narritive)
    if isinstance(image_input, dict):  # if upload from sketcher_input, input contains image and mask
        image_input = image_input['background']
        
    if isinstance(image_input, str):
        image_input = Image.open(io.BytesIO(base64.b64decode(image_input)))
    elif isinstance(image_input, bytes):
        image_input = Image.open(io.BytesIO(image_input))

    click_state = [[], [], []]
    image_input = image_resize(image_input, res=1024)

    model = build_caption_anything_with_models(
        args,
        api_key="",
        captioner=shared_captioner,
        sam_model=shared_sam_model,
        ocr_reader=shared_ocr_reader,
        session_id=iface.app_id
    )
    model.segmenter.set_image(image_input)
    image_embedding = model.image_embedding
    original_size = model.original_size
    input_size = model.input_size

    print('upload_callback: add caption to chatGPT memory')
    new_image_path = get_new_image_name('chat_image', func_name='upload')
    image_input.save(new_image_path)
    paragraph = get_gpt_response(openai_api_key, new_image_path,f"What's going on in this picture? in {language}")
    if task_type=="task 3":
        name="Along the River During the Qingming Festival"
        artist="Zhang Zeduan"
        year="12th century (Song Dynasty)"
        material="Chinese painting"
        gender="male"
    
    elif task_type=="task 1":
        name ="The Ambassadors"
        artist ="Hans Holbein the Younger"
        year = "1533 (Northern Renaissance)"
        material="Realism"
        gender = "male"
            
    elif task_type=="task 2":
        name = "The Football Players"
        artist= "Albert Gleizes"
        year= "1912 (Cubism)"
        material="Cubism"
        gender= "male"
        
    else:
        parsed_data = get_gpt_response(openai_api_key, new_image_path,"Please provide the name, artist, year of creation (including the art historical period), and painting style used for this painting. Return the information in dictionary format without any newline characters. Format as follows: { \"name\": \"Name of the painting\", \"artist\": \"Name of the artist\", \"year\": \"Year of creation (Art historical period)\", \"style\": \"Painting style used in the painting\"}")
        print(parsed_data)
        parsed_data = json.loads(parsed_data.replace("'", "\""))
        name, artist, year, material= parsed_data["name"],parsed_data["artist"],parsed_data["year"], parsed_data["style"]
        gender="male"
        gender=gender.lower()
                        
    if language=="English":
        if PromptTemplates.NARRATIVE_MAPPING[narritive]==0 :
            msg=f"🤖 Hi, I am EyeSee. Let's explore this painting '{name}' together. You can click on the area you're interested in and choose from four types of information: Description, Analysis, Interpretation, and Judgment. Based on your selection, I will provide you with the relevant information."
            
        elif PromptTemplates.NARRATIVE_MAPPING[narritive]==1:
            msg=f"🧑‍🎨 Hello, I am the {artist}. Welcome to explore my painting, '{name}'. You can click on the area you're interested in and choose from four types of information: Description, Analysis, Interpretation, and Judgment. Based on your selection, I will provide you with the relevant insights and thoughts behind my creation."
           
        elif PromptTemplates.NARRATIVE_MAPPING[narritive]==2:
            msg=f"🎨 Hello, Let's explore this painting '{name}' together. You can click on the area you're interested in and choose from four types of information: Description, Analysis, Interpretation, and Judgment. Based on your selection, I will provide you with relevant insights and thoughts from the perspective of the objects within the painting"
            
    elif language=="Chinese":
        if PromptTemplates.NARRATIVE_MAPPING[narritive]==0:
            msg=f"🤖 你好，我是 EyeSee。让我们一起探索这幅画《{name}》。你可以点击你感兴趣的区域，并选择四种信息类型之一：描述、分析、解读和评判。根据你的选择，我会为你提供相关的信息。"
            
        elif PromptTemplates.NARRATIVE_MAPPING[narritive]==1:
            msg=f"🧑‍🎨 你好，我是{artist}。欢迎探索我的画作《{name}》。你可以点击你感兴趣的区域，并选择四种信息类型之一：描述、分析、解读和评判。根据你的选择，我会为你提供我的创作背后的相关见解和想法。"
            
        elif PromptTemplates.NARRATIVE_MAPPING[narritive]==2:
            msg=f"🎨 你好，让我们一起探索这幅画《{name}》。你可以点击你感兴趣的区域，并选择四种信息类型之一：描述、分析、解读和评判。根据你的选择，我会从画面上事物的视角为你提供相关的见解和想法。"
            
    
    state = [(msg,None)]
    log_state += [(name,None)]
    log_state=log_state+[(paragraph,None)]
    log_state=log_state+[(narritive,None)]
    log_state=log_state+state
    log_state = log_state + [("%% basic information %%", None)]
    read_info=emoji.replace_emoji(msg,replace="") 
    
    history=[]
    history.append({"role": "assistant", "content": paragraph+msg})
    
    audio_output = None
    if autoplay:
        audio_output = await texttospeech(read_info, language,gender)

                                                                                                                                                                                                                                                                                         

    return [state, state, image_input, click_state]+[image_input]*1 + [image_embedding, \
        original_size, input_size] + [f"Name: {name}", f"Artist: {artist}", f"Year: {year}", f"Style: {material}"]*1 + [paragraph,artist, gender,new_image_path,log_state,history,audio_output]




def inference_click(image_input, point_prompt, click_mode, enable_wiki, language, sentiment, factuality,
                    length, image_embedding, state, click_state, original_size, input_size, text_refiner,
                    out_state, click_index_state, input_mask_state, input_points_state, input_labels_state, evt: gr.SelectData):
    click_index = evt.index

    if point_prompt == 'Positive':
        coordinate = "[[{}, {}, 1]]".format(str(click_index[0]), str(click_index[1]))
    else:
        coordinate = "[[{}, {}, 0]]".format(str(click_index[0]), str(click_index[1]))

    prompt = get_click_prompt(coordinate, click_state, click_mode)
    input_points = prompt['input_point']
    input_labels = prompt['input_label']

    controls = {'length': length,
                'sentiment': sentiment,
                'factuality': factuality,
                'language': language}

    model = build_caption_anything_with_models(
        args,
        api_key="",
        captioner=shared_captioner,
        sam_model=shared_sam_model,
        ocr_reader=shared_ocr_reader,
        text_refiner=text_refiner,
        session_id=iface.app_id
    )

    model.setup(image_embedding, original_size, input_size, is_image_set=True)
    
    enable_wiki = True if enable_wiki in ['True', 'TRUE', 'true', True, 'Yes', 'YES', 'yes'] else False
    out = model.inference(image_input, prompt, controls, disable_gpt=True, enable_wiki=enable_wiki, verbose=True, args={'clip_filter': False})[0]
    # state = state + [("You've selected image point at {}, ".format(prompt["input_point"]), None)]
    
    if language=="English":
        if prompt["input_label"][-1]==1:
            msg="You've added an area at {}. ".format(prompt["input_point"][-1])
        else:
            msg="You've removed an area at {}. ".format(prompt["input_point"][-1])
    else:
        if prompt["input_label"][-1]==1:
            msg="你添加了在 {} 的区域。 ".format(prompt["input_point"][-1])
        else:
            msg="你删除了在 {} 的区域。 ".format(prompt["input_point"][-1])

    state = state + [(msg, None)]
        
    input_mask = np.array(out['mask'].convert('P'))
    image_input_nobackground = mask_painter(np.array(image_input), input_mask,background_alpha=0)

    click_index_state = click_index
    input_mask_state = input_mask
    input_points_state = input_points
    input_labels_state = input_labels
    out_state = out  

    new_crop_save_path = get_new_image_name('chat_image', func_name='crop')
    Image.open(out["crop_save_path"]).save(new_crop_save_path)
    print("new crop save",new_crop_save_path)

    return state, state, click_state, image_input_nobackground, click_index_state, input_mask_state, input_points_state, input_labels_state, out_state,new_crop_save_path,image_input_nobackground
    

async def submit_caption(naritive, state,length, sentiment, factuality, language, 
                   out_state, click_index_state, input_mask_state, input_points_state, input_labels_state,
                   autoplay,paragraph,focus_type,openai_api_key,new_crop_save_path, gender,log_state,history):

    
    focus_value = PromptTemplates.FOCUS_MAP[focus_type]
    click_index = click_index_state
    
    print("click_index",click_index)
    print("input_points_state",input_points_state)
    print("input_labels_state",input_labels_state)
        
    prompt=generate_prompt(focus_type,paragraph,length,sentiment,factuality,language, naritive)
    print("log state",log_state[-1])
    if log_state[-1][0] is None or not log_state[-1][0].startswith("%%"):
        log_state =  log_state + [("No like/dislike", None)] 
        log_state =  log_state + [("%% user interaction %%",None)]
        
    log_state =  log_state + [("Selected image point: {}, Input label: {}".format(input_points_state, input_labels_state), None)] 
    
    print("Prompt:", prompt)
    print("click",click_index)
    log_state = log_state + [(naritive, None)]
    

    if not args.disable_gpt:
        print("new crop save",new_crop_save_path)
        focus_info=get_gpt_response(openai_api_key,new_crop_save_path,prompt)
        if focus_info.startswith('"') and focus_info.endswith('"'):
            focus_info=focus_info[1:-1]
        focus_info=focus_info.replace('#', '')
        # state = state + [(None, f"Wiki: {paragraph}")]
        if language=="English":
            user_query=query_focus_en[focus_value]
           
        else:
            user_query=query_focus_zh[focus_value]

        state = state + [(user_query, f"{focus_info}")]
        log_state = log_state + [(user_query, None)]
        log_state = log_state + [(None, f"{focus_info}")]
        
        # save history 
        history.append({"role": "user", "content": user_query})
        history.append({"role": "assistant", "content": focus_info})
        
        print("new_cap",focus_info)
        read_info = re.sub(r'[#[\]!*]','',focus_info)
        read_info = emoji.replace_emoji(read_info,replace="")    
        print("read info",read_info)
        gender="male"

        try:
            if autoplay==False:
                return state, state, click_index_state, input_mask_state, input_points_state, input_labels_state, out_state, None,log_state,history
            
            audio_output = await texttospeech(read_info, language,gender)
            print("done")
            # return state, state, refined_image_input, click_index_state, input_mask_state, input_points_state, input_labels_state, out_state, waveform_visual, audio_output
            return state, state, click_index_state, input_mask_state, input_points_state, input_labels_state, out_state, audio_output,log_state,history

        except Exception as e:
            state = state + [(None, f"Error during TTS prediction: {str(e)}")]
            print(f"Error during TTS prediction: {str(e)}")
            # return state, state, refined_image_input, click_index_state, input_mask_state, input_points_state, input_labels_state, out_state, None, None
            return state, state, click_index_state, input_mask_state, input_points_state, input_labels_state, out_state, audio_output,log_state,history

    else:
        state = state + [(None, f"Error during TTS prediction: {str(e)}")]
        print(f"Error during TTS prediction: {str(e)}")
        return state, state, click_index_state, input_mask_state, input_points_state, input_labels_state, out_state, None,None,log_state,history




def export_chat_log(chat_state,log_list,narrative):
    try:
        chat_log=""
        if not chat_state:
            return None
        for entry in chat_state:
            user_message, bot_response = entry
            if user_message and bot_response:
                chat_log += f"User: {user_message}\nBot: {bot_response}\n"
            elif user_message and user_message.startswith("%%"):
                chat_log += f"{user_message}\n"
            elif user_message:
                chat_log += f"User: {user_message}\n"
                chat_log += f"///// \n"
            elif bot_response:
                chat_log += f"Bot: {bot_response}\n"
                chat_log += f"///// \n"    
        
        print("export log...")
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{current_time}_{narrative}.txt"
        file_path = os.path.join(os.getcwd(), file_name)  # Save to the current working directory
        
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(chat_log)
        
        print(file_path)
            
        log_list.append(file_path)
        return log_list,log_list
    except Exception as e:
        print(f"An error occurred while exporting the chat log: {e}")
        return None,None

    
async def get_recommendation(new_crop, image_path, openai_api_key, language, autoplay, length, 
                           log_state, sort_score, narrative, state, recommend_type, artist, 
                           recommended_path):
    
    prompt = recommender.generate_recommendation_prompt(
        recommend_type=recommend_type,
        narrative=narrative,
        language=language,
        length=length,
        artist=artist
    )
    
    image_paths = [new_crop, recommended_path] if recommend_type == "Item" else [image_path, recommended_path]
    
    result = get_gpt_response(openai_api_key, image_paths, prompt)
    print("recommend result", result)
    
    log_message = (
        "User wants to know object recomendation reason" 
        if recommend_type == "Item" 
        else "User wants to know style recomendation reason"
    )
    
    state += [(None, f"{result}")]
    log_state += [(log_message, None)]
    log_state = log_state + [(narrative, None)]
    log_state = log_state + [(f"image sort ranking {sort_score}", None)]
    log_state = log_state + [(None, f"{result}")]

    read_info = re.sub(r'[#[\]!*]', '', result)
    read_info = emoji.replace_emoji(read_info, replace="")
    print("associate", read_info)

    audio_output = None
    if autoplay:
        audio_output = await texttospeech(read_info, language)
        
    return state, state, audio_output, log_state, None, gr.update(value=[])

def change_naritive(session_type,image_input, state, click_state, paragraph, origin_image,narritive,task_instruct,gallery_output,style_gallery_result,reco_reasons,language="English"):
    if session_type=="Session 1":
        return None, [], [], [[], [], []], "", None, None, [], [],[],[],gr.update(value="Preview"),None
    else:
        if language=="English":
            if narritive=="Narrator" :
                state += [
                (
                    None,
                    f"🤖 Hi, I am EyeSee. Let's explore this painting together."
                )
                ]
            elif narritive=="Artist":
                state += [
                (
                    None,
                    f"🧑‍🎨 Let's delve into it from the perspective of the artist."
                )
                ]
            elif narritive=="In-Situ":
                state += [
                (
                    None,
                    f"🎨 Let's delve into it from the perspective of the objects depicted in the scene."
                )
                ]
        elif language=="Chinese":
            if narritive=="Narrator" :
                state += [
                    (
                        None,
                        "🤖 让我们从第三方视角一起探索这幅画吧。"
                    )
                ]
            elif narritive == "Artist":
                state += [
                (
                    None,
                    "🧑‍🎨 让我们从艺术家的视角深入探索这幅画。"
                )
            ]
            elif narritive == "In-Situ":
                state += [
                (
                    None,
                    "🎨 让我们从画面中事物的视角深入探索这幅画。"
                )
            ]


        return image_input, state, state, click_state, paragraph, origin_image,task_instruct,gallery_output,style_gallery_result,reco_reasons,reco_reasons,gr.update(value="Preview"),None


def print_like_dislike(x: gr.LikeData,state,log_state):
    print(x.index, x.value, x.liked)
    if x.liked == True:
        print("liked")
        log_state=log_state+[(f"User liked this message", None)]
        state = state + [(None, f"Liked Received 👍")]
    else:
        log_state=log_state+[(f"User disliked this message", None)]
        state = state + [(None, f"Disliked Received 👎")]
    log_state+=[("%% user interaction %%", None)]
    return log_state,state

def get_recommendationscore(index,score,log_state):
    log_state+=[(f"{index} : {score}",None)]
    log_state+=[("%% recommendation %%",None)]
    return log_state
    

add_icon_path="assets/icons/plus-square-blue.png"
minus_icon_path="assets/icons/minus-square.png"  
  
def toggle_icons_and_update_prompt(point_prompt):
    new_prompt = "Negative" if point_prompt == "Positive" else "Positive"
    new_add_icon = "assets/icons/plus-square-blue.png" if new_prompt == "Positive" else "assets/icons/plus-square.png"
    new_add_css = "tools_button_clicked" if new_prompt == "Positive" else "tools_button"
    new_minus_icon = "assets/icons/minus-square.png" if new_prompt == "Positive" else "assets/icons/minus-square-blue.png"
    new_minus_css= "tools_button" if new_prompt == "Positive" else "tools_button_clicked"

    return new_prompt, gr.update(icon=new_add_icon,elem_classes=new_add_css), gr.update(icon=new_minus_icon,elem_classes=new_minus_css)
        
    


with open('styles.css', 'r') as file:
    css = file.read()
def create_ui():
    print(6)
    title = """<p><h1 align="center">EyeSee Anything in Art</h1></p>
    """
    description = """<p>Gradio demo for EyeSee Anything in Art, image to dense captioning generation with various language styles. To use it, simply upload your image, or click one of the examples to load them. """

    examples = [
        ["assets/test_images/1.The Ambassadors.jpg","assets/test_images/task1.jpg","task 1"],
        ["assets/test_images/2.Football Players.jpg","assets/test_images/task2.jpg","task 2"],
        ["assets/test_images/3-square.jpg","assets/test_images/task3.jpg","task 3"]]

    with gr.Blocks(
            css=css,
            theme=gr.themes.Base()
    ) as iface:
        #display in the chatbox 
        state = gr.State([])
        # export in log
        log_state=gr.State([])
        # history log for gpt
        history_log=gr.State([])
        
        out_state = gr.State(None)
        click_state = gr.State([[], [], []])
        origin_image = gr.State(None)
        image_embedding = gr.State(None)
        text_refiner = gr.State(None)
        original_size = gr.State(None)
        input_size = gr.State(None)
        paragraph = gr.State("")
        aux_state = gr.State([])
        click_index_state = gr.State((0, 0))
        input_mask_state = gr.State(np.zeros((1, 1)))
        input_points_state = gr.State([])
        input_labels_state = gr.State([])
        new_crop_save_path = gr.State(None)
        image_input_nobackground = gr.State(None)
        artist=gr.State(None)
        gr.Markdown(title)
        gr.Markdown(description)
        point_prompt = gr.State("Positive") 
        log_list=gr.State([])
        gender=gr.State('female')
        image_path=gr.State('')
        pic_index=gr.State(None)
        recomended_state=gr.State([])     
        recomended_path=gr.State(None)
        recomended_type=gr.State(None)
        
        with gr.Row():  
            
            with gr.Column(scale=6):
                with gr.Column(visible=False) as modules_not_need_gpt:

                    with gr.Row():
                        naritive = gr.Radio(
                        choices=["Narrator", "Artist","In-Situ"],
                        value="Narrator",
                        label="Select Mode",
                        scale=5,
                        interactive=True) 
                        
                        add_button = gr.Button(value="Extend Area", interactive=True,elem_classes="tools_button_add",icon=add_icon_path)
                        minus_button = gr.Button(value="Remove Area", interactive=True,elem_classes="tools_button",icon=minus_icon_path)
                        clear_button_click = gr.Button(value="Reset", interactive=True,elem_classes="tools_button",icon="assets/icons/recycle.png") 
                                                
                        auto_play = gr.Checkbox(
                        label="Check to autoplay audio", value=False, elem_classes="custom-autoplay",visible=False)
                        output_audio = gr.HTML(
                            label="Synthesised Audio", elem_classes="custom-output", visible=False)
                    

                    with gr.Row():
                        with gr.Column(scale=1,min_width=50,visible=False) as instruct:
                            task_instuction=gr.Image(type="pil", interactive=False, elem_classes="task_instruct",height=650,label="Instruction")        
                        with gr.Column(scale=6):
                            with gr.Tab("Click") as click_tab:
                                with gr.Row():
                                    with gr.Column(scale=10,min_width=600):
                                        image_input = gr.Image(type="pil", interactive=True, elem_classes="image_upload",height=650)
                                        example_image = gr.Image(type="pil", interactive=False, visible=False)
                                    # example_image_click = gr.Image(type="pil", interactive=False, visible=False)
                                    # the tool column
                                    with gr.Column(scale=1,elem_id="tool_box",min_width=80):
                                        name_label = gr.Button(value="Name: ",elem_classes="info_btn")
                                        artist_label = gr.Button(value="Artist: ",elem_classes="info_btn_interact")
                                        year_label = gr.Button(value="Year: ",elem_classes="info_btn_interact")
                                        material_label = gr.Button(value="Style: ",elem_classes="info_btn")

                                        focus_d = gr.Button(value="Describe",interactive=True,elem_classes="function_button") 
                                        focus_da = gr.Button(value="D+Analysis",interactive=True,elem_classes="function_button") 
                                        focus_dai = gr.Button(value="DA+Interprete",interactive=True,elem_classes="function_button") 
                                        focus_dda = gr.Button(value="Judge",interactive=True,elem_classes="function_button")

                                        recommend_btn = gr.Button(value="Recommend",interactive=True,elem_classes="function_button_rec")
                             

                        
                                                                    
                    with gr.Column(visible=False,scale=4) as modules_need_gpt1:
                        with gr.Row(visible=False):
                            sentiment = gr.Radio(
                                choices=["Positive", "Natural", "Negative"],
                                value="Natural",
                                label="Sentiment",
                                interactive=True,
                            )

                            factuality = gr.Radio(
                                choices=["Factual", "Imagination"],
                                value="Factual",
                                label="Factuality",
                                interactive=True,
                            )

                            enable_wiki = gr.Radio(
                                choices=["Yes", "No"],
                                value="No",
                                label="Expert",
                                interactive=True)
                with gr.Column(visible=True) as modules_not_need_gpt3:
                    gr.Examples(
                examples=examples,
                inputs=[example_image],
            )
                

            with gr.Column(scale=4):                    
                with gr.Column(visible=True) as module_key_input:
                    openai_api_key = gr.Textbox(
                        value="",
                        placeholder="Input openAI API key",
                        show_label=False,
                        label="OpenAI API Key",
                        lines=1,
                        type="password")
                    with gr.Row():
                        enable_chatGPT_button = gr.Button(value="Run with ChatGPT", interactive=True, variant='primary')

                with gr.Column(visible=False) as module_notification_box:
                    notification_box = gr.Textbox(lines=1, label="Notification", max_lines=5, show_label=False)

                with gr.Column(visible=False) as modules_not_need_gpt2: 
                            with gr.Blocks():
                                chatbot = gr.Chatbot(label="Chatbox", elem_classes="chatbot",likeable=True,height=750,bubble_full_width=False)
                                with gr.Column() as modules_need_gpt3:
                                    chat_input = gr.MultimodalTextbox(interactive=True, file_types=[".txt"], placeholder="Message EyeSee...", show_label=False)
                                    with gr.Row(): 
                                        clear_button_text = gr.Button(value="Clear Chat", interactive=True)
                                        export_button = gr.Button(value="Export Chat Log", interactive=True, variant="primary")
                            with gr.Row(visible=False):
                                with gr.Column():                                                                          
                                    with gr.Row():
                                        click_mode = gr.Radio(
                                            choices=["Continuous", "Single"],
                                            value="Continuous",
                                            label="Clicking Mode",
                                            scale=5,
                                            interactive=True)
                
        
        with gr.Row():
            with gr.Column(scale=6):
                with gr.Row():
                    with gr.Column(visible=False) as recommend:

                        gallery_result = gr.Gallery(
                        label="Object-based Recommendation",
                        height="auto",
                        columns=2,
                        interactive=False)
                    
                        style_gallery_result = gr.Gallery(
                        label="Style-based Recommendation",
                        height="auto",
                        columns=2,
                        interactive=False)
                    with gr.Column(scale=3,visible=False) as reco_preview:
                        selected_image = gr.Image(label="Selected Image", interactive=False)
                    
                    sort_rec = gr.Radio(
                            choices=[1,2,3,4,5,6,7],
                            label="Score",
                            visible=False,
                            interactive=True,info="Please sort the recommendation artwork")    
                    
                    recommend_type = gr.Radio(
                            choices=["Preview","Reasons"],
                            label="Information Type",
                            value="Preview",
                            interactive=True,visible=False)   
                    

            with gr.Column(scale=4,visible=False) as reco_reasons:
                recommend_bot = gr.Chatbot(label="Recommend Reasons", elem_classes="chatbot",height=600)
                recommend_score = gr.Radio(
                            choices=[1,2,3,4,5,6,7],
                            label="Score",
                            interactive=True,info='Please score the recommendation reasons')             
        
        with gr.Row():
            task_type = gr.Textbox(visible=False)
            gr.Examples(
        examples=examples,
        inputs=[example_image,task_instuction,task_type],
        )                
   
        with gr.Row(visible=False) as export:
            chat_log_file = gr.File(label="Download Chat Log",scale=5)
        
        with gr.Row(elem_id="top_row",visible=False) as top_row:  
            session_type = gr.Dropdown(
            ["Session 1","Session 2"],
            value="Session 1", label="Task", interactive=True, elem_classes="custom-language"
        )
            language = gr.Dropdown(
            ['English', 'Chinese', 'French', "Spanish", "Arabic", "Portuguese", "Cantonese"],
            value="English", label="Language", interactive=True, elem_classes="custom-language"
        )
            length = gr.Slider(
                                minimum=60,
                                maximum=120,
                                value=80,
                                step=1,
                                interactive=True,
                                label="Generated Caption Length",
                            )

        recommend_btn.click(
            fn=recommender.infer,
            inputs=[new_crop_save_path,image_path,state,language,task_type],
            outputs=[gallery_result,style_gallery_result,chatbot,state]
            )

        gallery_result.select(
            recommender.item_associate,
            inputs=[new_crop_save_path,openai_api_key,language,auto_play,length,log_state,sort_rec,naritive,recomended_state],
            outputs=[recommend_bot,recomended_state,output_audio,log_state,pic_index,recommend_score,selected_image,recomended_path, recomended_type],
            
            
        )
        
        style_gallery_result.select(
            recommender.style_associate,
            inputs=[image_path,openai_api_key,language,auto_play,length,log_state,sort_rec,naritive,recomended_state,artist_label],
            outputs=[recommend_bot,recomended_state,output_audio,log_state,pic_index,recommend_score,selected_image,recomended_path,recomended_type])
        
        selected_image.select(
            get_recommendation,
            inputs=[new_crop_save_path,image_path, openai_api_key,language,auto_play,length,log_state,sort_rec,naritive,recomended_state,recomended_type,artist_label,recomended_path],
            outputs=[recommend_bot,recomended_state,output_audio,log_state,pic_index,recommend_score])
        
        
        chatbot.like(print_like_dislike, inputs=[state,log_state], outputs=[log_state,chatbot])
              
        
        recommend_score.select(
            get_recommendationscore,
            inputs=[pic_index,recommend_score,log_state],
            outputs=[log_state],
        )

   
        openai_api_key.submit(init_openai_api_key, inputs=[openai_api_key],
                              outputs=[export, modules_need_gpt1, modules_need_gpt3, modules_not_need_gpt,
                                       modules_not_need_gpt2, module_key_input ,module_notification_box, text_refiner, notification_box,top_row,recommend,reco_reasons,instruct,modules_not_need_gpt3,sort_rec,reco_preview])
        enable_chatGPT_button.click(init_openai_api_key, inputs=[openai_api_key],
                                    outputs=[export,modules_need_gpt1, modules_need_gpt3,
                                             modules_not_need_gpt,
                                             modules_not_need_gpt2,module_key_input,module_notification_box, text_refiner, notification_box,top_row,recommend,reco_reasons,instruct,modules_not_need_gpt3,sort_rec,reco_preview])
        
        artist_label.click(
            get_artistinfo,
            inputs=[artist_label,openai_api_key,state,language,auto_play,length,log_state],
            outputs=[chatbot,state,output_audio,log_state]
        )

        year_label.click(
            get_yearinfo,
            inputs=[year_label,openai_api_key,state,language,auto_play,length,log_state],
            outputs=[chatbot,state,output_audio,log_state]
        )

        def reset_and_add(origin_image):
            new_prompt = "Positive"
            new_add_icon = "assets/icons/plus-square-blue.png"
            new_add_css = "tools_button_clicked"
            new_minus_icon = "assets/icons/minus-square.png"
            new_minus_css= "tools_button" 
            return [[],[],[]],origin_image, new_prompt, gr.update(icon=new_add_icon,elem_classes=new_add_css), gr.update(icon=new_minus_icon,elem_classes=new_minus_css)
            
        clear_button_click.click(
            reset_and_add,
            [origin_image],
            [click_state, image_input,point_prompt,add_button,minus_button],
            queue=False,
            show_progress=False
        )


        clear_button_text.click(
            lambda: ([], [], [[], [], [], []],[]),
            [],
            [chatbot, state, click_state,history_log],
            queue=False,
            show_progress=False
        )

        
        image_input.clear(
            lambda: (None, [], [], [[], [], []], "", None, []),
            [],
            [image_input, chatbot, state, click_state, paragraph, origin_image,history_log],
            queue=False,
            show_progress=False
        )


          
        
        image_input.upload(upload_callback, [image_input, state, log_state,task_type,openai_api_key,language,naritive,history_log,auto_play,session_type],
                           [chatbot, state, origin_image, click_state, image_input,image_embedding, original_size, input_size,name_label,artist_label,year_label,material_label,\
                            paragraph,artist,gender,image_path,log_state,history_log,output_audio])
        
        
        
        chat_input.submit(chat_input_callback, [chat_input,state, aux_state,language,auto_play,gender,openai_api_key,image_path,log_state,history_log,naritive],
                          [chatbot, state, aux_state,output_audio,log_state,history_log])

        chat_input.submit(lambda: {"text": ""}, None, chat_input)

        example_image.change(upload_callback, [example_image, state, log_state, task_type, openai_api_key,language,naritive,history_log,auto_play,session_type],
                             [chatbot, state, origin_image, click_state, image_input, image_embedding, original_size, input_size,name_label,artist_label,year_label,material_label,\
                            paragraph,artist,gender,image_path, log_state,history_log,output_audio])

        example_image.change(
            lambda:([],[],[],None,[],gr.update(value="Preview"),None), 
            [],
            [gallery_result,style_gallery_result,recommend_bot,new_crop_save_path,chatbot,recommend_type,selected_image])
        
        image_input.select(
            inference_click,
            inputs=[
                origin_image, point_prompt, click_mode, enable_wiki, language, sentiment, factuality, length,
                image_embedding, state, click_state, original_size, input_size, text_refiner,
                out_state, click_index_state, input_mask_state, input_points_state, input_labels_state
            ],
            outputs=[chatbot, state, click_state, image_input, click_index_state, input_mask_state, input_points_state, input_labels_state, out_state,new_crop_save_path,image_input_nobackground],
            show_progress=False, queue=True
        )


        focus_d.click(
            submit_caption,
            inputs=[
        naritive, state,length, sentiment, factuality, language, 
        out_state, click_index_state, input_mask_state, input_points_state, input_labels_state, auto_play, paragraph,focus_d,openai_api_key,new_crop_save_path,gender,log_state,history_log
    ],
            outputs=[
                chatbot, state, click_index_state, input_mask_state, input_points_state, input_labels_state, out_state,output_audio,log_state,history_log
            ],
            show_progress=True,
            queue=True
        )
        

        

        
        focus_da.click(
        submit_caption,
        inputs=[
        naritive,state,length, sentiment, factuality, language, 
        out_state, click_index_state, input_mask_state, input_points_state, input_labels_state,auto_play, paragraph,focus_da,openai_api_key,new_crop_save_path,gender,log_state,
        history_log
        ],
        outputs=[
                chatbot, state, click_index_state, input_mask_state, input_points_state, input_labels_state, out_state,output_audio,log_state,history_log
            ],
        show_progress=True,
        queue=True
        )

        
        focus_dai.click(
        submit_caption,
        inputs=[
        naritive,state,length, sentiment, factuality, language, 
        out_state, click_index_state, input_mask_state, input_points_state, input_labels_state,
        auto_play, paragraph,focus_dai,openai_api_key,new_crop_save_path,gender,log_state,history_log
        ],
        outputs=[
            chatbot, state, click_index_state, input_mask_state, input_points_state, input_labels_state, out_state,output_audio,log_state,history_log
        ],
        show_progress=True,
        queue=True
        )
        
        
        focus_dda.click(
        submit_caption,
        inputs=[
        naritive,state,length, sentiment, factuality, language, 
        out_state, click_index_state, input_mask_state, input_points_state, input_labels_state,
        auto_play, paragraph,focus_dda,openai_api_key,new_crop_save_path,gender,log_state,history_log
        ],
        outputs=[
            chatbot, state, click_index_state, input_mask_state, input_points_state, input_labels_state, out_state,output_audio,log_state,history_log
        ],
        show_progress=True,
        queue=True
        )
        
        add_button.click(
            toggle_icons_and_update_prompt,
            inputs=[point_prompt],
            outputs=[point_prompt,add_button,minus_button],
            show_progress=True,
            queue=True
            
        )
        
        minus_button.click(
            toggle_icons_and_update_prompt,
            inputs=[point_prompt],
            outputs=[point_prompt,add_button,minus_button],
            show_progress=True,
            queue=True
            
        )
                
        
        export_button.click(
            export_chat_log,
            inputs=[log_state,log_list,naritive],
            outputs=[chat_log_file,log_list],
            queue=True
        )
        
        naritive.change(
            change_naritive,
            [session_type, image_input, state, click_state, paragraph, origin_image,naritive,
             task_instuction,gallery_result,style_gallery_result,recomended_state,language],
            [image_input, chatbot, state, click_state, paragraph, origin_image,task_instuction,gallery_result,style_gallery_result,recomended_state,recommend_bot,recommend_type,selected_image],
            queue=False,
            show_progress=False
        
        )
        def change_session():
            instruction=Image.open('test_images/task4.jpg')
            return None, [], [], [[], [], []], "", None, [],[],instruction,"task 4",[],[],[]
        
        session_type.change(
            change_session,
            [],
            [image_input, chatbot, state, click_state, paragraph, origin_image,history_log,log_state,task_instuction,task_type,gallery_result,style_gallery_result,recommend_bot]
        )

        return iface


if __name__ == '__main__':
    print("main")
    iface = create_ui()
    iface.queue(api_open=False, max_size=10)
    iface.launch(server_name="0.0.0.0")