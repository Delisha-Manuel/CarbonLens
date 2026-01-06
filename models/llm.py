import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_advice(user_data: dict, top_features: list) -> str:
    top_features_str = "\n".join([f"- {name}: {value:.2f} kg CO₂" for name, value in top_features])

    prompt = f"""YOU are an environmental advisor. PROVIDE 3 ACTIONABLE TIPS for me to REDUCE MY CARBON FOOTPRINT.
    
    TOP CONTRIBUTORS to my carbon footprint: {top_features_str}
        
    Here is some information about me:
        My diet: {user_data.get('diet')}
        How often I shower: {user_data.get('shower')}
        My source of energy for heat: {user_data.get('heating')}
        My transportation preference: {user_data.get('transport')}
        The type of fuel my vehicle uses: {user_data.get('vehicle')}
        How far I travel in my vehicle in km per month: {user_data.get('distance')}
        How many flights I took last month: {user_data.get('air')}
        How much I spend on groceries in dollars per month: {user_data.get('grocery')}
        How many waste bags I use per week: {user_data.get('waste_count')}
        The number of clothes I purchase per month: {user_data.get('clothes')}
        The number of hours I spend on the internet per day: {user_data.get('internet')}
        The number of hours I spend on my TV or personal computer each day: {user_data.get('tv_pc')}
        What I recycle: {', '.join(user_data.get('recycling', []))}
        What I use to cook food: {', '.join(user_data.get('cooking', []))}"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print("DEBUG: started generation")
    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    raw_output = generated_text[len(prompt):].strip()
    raw_output = remove_quotes(raw_output)
    
    print("DEBUG: finished generation")

    return raw_output


def remove_quotes(text: str) -> str:
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1].strip()
    return text
