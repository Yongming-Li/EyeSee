class PromptTemplates:
    FOCUS_MAP = {                                                     
        "Describe": 0, 
        "D+Analysis": 1, 
        "DA+Interprete": 2,
        "Judge": 3
    }

    NARRATIVE_MAPPING = {
        "Narrator": 0,
        "Artist": 1,
        "In-Situ": 2
    }

    ANALYSIS_PROMPTS = [
        [
            'Wiki_caption: {Wiki_caption}, you have to help me understand what is about the selected object and list one fact (describes the selected object but does not include analysis) as markdown outline with appropriate emojis that describes what you see according to the image and wiki caption. Each point listed is to be in {language} language, with a response length of about {length} words.',
            'Wiki_caption: {Wiki_caption}, you have to help me understand what is about the selected object and list one fact and one analysis as markdown outline with appropriate emojis that describes what you see according to the image and wiki caption.  Each point listed is to be in {language} language, with a response length of about {length} words.',
            'Wiki_caption: {Wiki_caption}, you have to help me understand what is about the selected object and list one fact and one analysis and one interpret as markdown outline with appropriate emojis that describes what you see according to the image and wiki caption. Each point listed is to be in {language} language, with a response length of about {length} words.',
            'Wiki_caption: {Wiki_caption}, You have to help me understand what is about the selected object and list one object judgement and one whole art judgement(how successful do you think the artist was?) as markdown outline with appropriate emojis that describes what you see according to the image and wiki caption. Each point listed is to be in {language} language, with a response length of about {length} words.'
        ],
        [
            "When generating the answer, you should tell others that you are the creator of this painting and generate the text in the tone and manner as if you are the creator of this painting. You have to help me understand what is about the selected object and list one fact (describes the selected object but does not include analysis) as markdown outline with appropriate emojis that describes what you see according to the image and {Wiki_caption}. Please generate the above points in the tone and manner as if you are the creator of this painting and start every sentence with I. Please generate the above points in the tone and manner as if you are the creator of this painting and start every sentence with I. Each point listed is to be in {language} language, with a response length of about {length} words.",
            "When generating the answer, you should tell others that you are the creator of this painting and generate the text in the tone and manner as if you are the creator of this painting. You have to help me understand what is about the selected object and list one fact and one analysis from art appreciation perspective as markdown outline with appropriate emojis that describes what you see according to the image and {Wiki_caption}. Please generate the above points in the tone and manner as if you are the creator of this painting and start every sentence with I. Please generate the above points in the tone and manner as if you are the creator of this painting and start every sentence with I. Each point listed is to be in {language} language, with a response length of about {length} words.",
            "When generating the answer, you should tell others that you are the creator of this painting and generate the text in the tone and manner as if you are the creator of this painting. You have to help me understand what is about the selected object and list one fact, one analysis, and one interpret from art appreciation perspective as markdown outline with appropriate emojis that describes what you see according to the image and {Wiki_caption}. Please generate the above points in the tone and manner as if you are the creator of this painting and start every sentence with I. Please generate the above points in the tone and manner as if you are the creator of this painting and start every sentence with I. Each point listed is to be in {language} language, with a response length of about {length} words.",
            "When generating the answer, you should tell others that you are one of the creators of these paintings and generate the text in the tone and manner as if you are the creator of the painting. According to image and wiki_caption {Wiki_caption}, You have to help me understand what is about the selected object and list one object judgement and one whole art judgement(how successful do you think the artist was?) as markdown outline with appropriate emojis that describes what you see according to the image and wiki caption. Please generate the above points in the tone and manner as if you are the creator of this painting and start every sentence with I. Each point listed is to be in {language} language, with a response length of about {length} words.",
        ],
        [
            'When generating answers, you should tell people that you are the object itself that was selected, and generate text in the tone and manner in which you are the object or the person. You have to help me understand what is about the selected object and list one fact (describes the selected object but does not include analysis) as markdown outline with appropriate emojis that describes what you see according to the image and {Wiki_caption}. Please generate the above points in the tone and manner as if you are the object and start every sentence with I. Please generate the above points in the tone and manner as if you are the object of this painting and start every sentence with I. Each point listed is to be in {language} language, with a response length of about {length} words.',
            'When generating answers, you should tell people that you are the object itself that was selected, and generate text in the tone and manner in which you are the object or the person. You have to help me understand what is about the selected object and list one fact and one analysis from art appreciation perspective as markdown outline with appropriate emojis that describes what you see according to the image and {Wiki_caption}. Please generate the above points in the tone and manner as if you are the object and start every sentence with I. Please generate the above points in the tone and manner as if you are the object of this painting and start every sentence with I. Each point listed is to be in {language} language, with a response length of about {length} words.',
            'When generating answers, you should tell people that you are the object itself that was selected, and generate text in the tone and manner in which you are the object or the person. You have to help me understand what is about the selected object and list one fact and one analysis from art appreciation perspective and one interpret as markdown outline with appropriate emojis that describes what you see according to the image and {Wiki_caption}. Please generate the above points in the tone and manner as if you are the object and start every sentence with I. Please generate the above points in the tone and manner as if you are the object of this painting and start every sentence with I.  Each point listed is to be in {language} language, with a response length of about {length} words.',
            'When generating answers, you should tell people that you are the object itself that was selected, and generate text in the tone and manner in which you are the object or the person. According to image and wiki_caption {Wiki_caption}, You have to help me understand what is about the selected object and list one object judgement and one whole art judgement(how successful do you think the artist was?) as markdown outline with appropriate emojis that describes what you see according to the image and wiki caption. Please generate the above points in the tone and manner as if you are the object of this painting and start every sentence with I. Each point listed is to be in {language} language, with a response length of about {length} words.',
        ]
        ]

    RECOMMENDATION_PROMPTS = [
    
    [
    '''
    First identify what the object of the first painting is, you save yourself as the parameter: {{object}}, do not need to tell me, the following will use the parameter. I want you to write the recommendation reason according to the following content, as a markdown outline with appropriate emojis that describe what you see according to the painting: 
    Recommendation reason: {{Recommendation based on {{object}} in the painting you saw earlier. Detailed analysis: Based on the recommendation reason and the relationship between the two paintings, explain why you recommend another painting. Please generate in three points. }}
    Each bullet point should be in {language} language, with a response length of about {length} words.
    ''',
    '''
    When generating answers, you should tell people that I am the creator of painting you were looking at earlier itself, and generate text in the tone and manner in which you are the creator of painting were looking at earlier. 

    First identify what the object of the first painting is, you save yourself as the parameter: {{object}}, do not need to tell me, the following will use the. I want you to write the recommendation reason according to the following content, as a markdown outline with appropriate emojis that describe what you see according to the painting: 

    Recommendation reason: {{I'm the creator of that painting you saw earlier. I'm an artist. and I'm recommending this painting based on the fact that the {{object}} I've drawn also appear in the painting you're looking at. }} Detailed analysis: Based on the recommendation reason and the relationship between the two paintings, explain why you recommend another painting. Please generate the three points in the tone and manner as if you are the creator of painting were looking at earlier and start every sentence with I.

    Each bullet point should be in {language} language, with a response length of about {length} words.

    ''',
    '''
    When generating answers, you should tell people that you are the object itself that was selected in the painting, and generate text in the tone and manner in which you are the object 

    First identify what the object of the first painting is, you save yourself as the parameter: {{object}}, do not need to tell me, the following will use the parameter. I want you to write the recommendation reason according to the following content, as a markdown outline with appropriate emojis that describe what you see according to the painting: 

    Recommendation reason: {{I'm the {{object}} in the painting you were looking at earlier, and I'm recommending this painting based on the fact that I'm also present in the one you're looking at.}} Detailed analysis: Based on the recommendation reason and the relationship between the two paintings, explain why you recommend another painting. Please generate the three points in the tone and manner as if you are the object of this painting and start every sentence with I. 

    Each bullet point should be in {language} language, with a response length of about {length} words.

    '''],
    
    [
    '''
    First identify what the name of the first painting is, you save yourself as the parameter: {{name}}, do not need to tell me, the following will use the parameter. I want you to write the recommendation reason according to the following content, as a markdown outline with appropriate emojis that describe what you see according to the painting: 
    Recommendation reason: {{Recommendation based on the painting {{name}}.Detailed analysis: Based on the recommendation reason and the relationship between the two paintings, explain why you recommend another painting. Please generate in three points.}} 
    Each bullet point should be in {language} language, with a response length of about {length} words.
    ''',
    '''
    When generating answers, you should tell people that I am the creator of painting you were looking at earlier itself, and generate text in the tone and manner in which you are the creator of painting were looking at earlier. 

    First identify what the creator of the first painting is, you save yourself as the parameter: {artist}, do not need to tell me, the following will use the parameter. I want you to write the recommendation reason according to the following content, as a markdown outline with appropriate emojis that describe what you see according to the painting: 

    Recommendation reason: {{I'm the creator of that painting you saw earlier, {artist}. I'm an artist. and I'm recommending this painting based on the fact that the painting you're looking at is similar to the one you just saw of me.}} Detailed analysis: Based on the recommendation reason and the relationship between the two paintings, explain why you recommend another painting. Please generate the three points in the tone and manner as if you are the creator of painting were looking at earlier and start every sentence with I.

    Each bullet point should be in {language} language, with a response length of about {length} words.

    ''',
    '''
    When generating answers, you should tell people that I am the painting you were looking at earlier itself, and generate text in the tone and manner in which you are the painting were looking at earlier. 

    First identify what the name of the first painting is, you save yourself as the parameter: {{name}}, do not need to tell me, the following will use the parameter. I want you to write the recommendation reason according to the following content, as a markdown outline with appropriate emojis that describe what you see according to the painting: 

    Recommendation reason: {{I'm the painting {{name}} you were looking at earlier, and I'm recommending this painting based on the fact that I'm similar to the one you're looking at.}} Detailed analysis: Based on the recommendation reason and the relationship between the two paintings, explain why you recommend another painting. Please generate the three points in the tone and manner as if you are the painting were looking at earlier and start every sentence with I. 

    Each bullet point should be in {language} language, with a response length of about {length} words.

    '''],
]   
