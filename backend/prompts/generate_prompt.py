from .prompt_templates import PromptTemplates

def generate_prompt(focus_type, paragraph, length, sentiment, factuality, language, narrative):
    mapped_value = PromptTemplates.FOCUS_MAP.get(focus_type, -1)
    narrative_value = PromptTemplates.NARRATIVE_MAPPING[narrative]

    controls = {
        'length': length,
        'sentiment': sentiment,
        'factuality': factuality,
        'language': language
    }

    if mapped_value != -1:
        prompt = PromptTemplates.ANALYSIS_PROMPTS[narrative_value][mapped_value].format(
            Wiki_caption=paragraph,
            length=controls['length'],
            sentiment=controls['sentiment'],
            language=controls['language']
        )
    else:
        prompt = "Invalid focus type."
    return prompt