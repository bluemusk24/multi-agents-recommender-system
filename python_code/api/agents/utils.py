# chatbot response from prompt_engr_tutorial.ipynb
def get_chatbot_response(client,model_name,messages,temperature=0):
    input_messages = []
    for message in messages:
        input_messages.append({"role": message["role"], "content": message["content"]})

    response = client.chat.completions.create(
        model=model_name,
        messages=input_messages,
        temperature=temperature,
        top_p=0.8,
        max_tokens=2000,
    ).choices[0].message.content
    
    return response

# embedding from prompt_engr_tutorial.ipynb
def get_embedding(embedding_client,model_name,text_input):
    output = embedding_client.embeddings.create(input = text_input,model=model_name)
    
    embedings = []
    for embedding_object in output.data:
        embedings.append(embedding_object.embedding)

    return embedings

# function to fix any json output errors
def double_check_json_output(client, model_name, json_string):
    prompt = f"""You are an AI assistant. Your job is to validate and correct JSON strings. 
    Follow these rules strictly:
    1. If the JSON string is correct, return it without any modifications.
    2. If the JSON string is invalid, correct it and return only the valid JSON.
    3. Ensure the JSON keys: "chain of thought", "decision", "message", "recommendation_type", "order", "step number" are enclosed in double quotes and that the structure is a proper JSON object.
    4. Do not add or remove extra information—just ensure the JSON is valid and includes all required fields.
    5. The first character of your response must be an open curly brace '{{', and the last character must be a closing curly brace '}}'.

    Here is the JSON string to validate and correct:

    ```{json_string}```"""
    
    messages = [{"role": "user", "content": prompt}]
    response = get_chatbot_response(client, model_name, messages)
    response = response.replace("`", "").strip()
    return response