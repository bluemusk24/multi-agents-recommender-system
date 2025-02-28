from dotenv import load_dotenv, find_dotenv
import os
import json
from copy import deepcopy
from .utils import get_chatbot_response, double_check_json_output
from openai import OpenAI

# initialize .env file
_ = load_dotenv(find_dotenv())


class GuardAgent():
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("RUNPOD_TOKEN"),
            base_url=os.getenv("RUNPOD_CHATBOT_URL"),
        )
        self.model_name = os.getenv("MODEL_NAME")
    
    # the get_response function from the agent_protocol.py
    def get_response(self, messages):
        messages = deepcopy(messages)
        
        system_prompt = """
            You are a helpful AI assistant for a coffee shop application that serves drinks and pastries.
            Your task is to determine whether the user is asking something relevant to the coffee shop or not.

            IMPORTANT: You must return exactly one JSON object. Do not return multiple responses.

            The user is allowed to:
            1. Ask questions about the coffee shop, like location, working hours, menu items, and coffee shop-related questions.
            2. Ask questions about menu items, including ingredients and more details about the item.
            3. Make an order.
            4. Ask for recommendations of what to buy.

            The user is NOT allowed to:
            1. Ask questions about anything unrelated to the coffee shop.
            2. Ask questions about the staff or how to make a specific menu item.

            Return exactly one JSON object in this format:

            For allowed inputs:
            {
                "chain of thought": "Provide reasoning that aligns the user's input with the allowed categories, explaining why the input is permissible.",
                "decision": "allowed",
                "message": "I'll help you with that."
            }

            For not allowed inputs:
            {
                "chain of thought": "Provide reasoning that aligns the user's input with the prohibited categories, explaining why the input is not permissible.",
                "decision": "not allowed",
                "message": "Sorry, I can't help with that. Can I help you with your order?"
            }

            Remember: Return EXACTLY ONE JSON response. Do not provide multiple responses or any text outside the JSON.
        """        

        input_messages = [{"role": "system", "content": system_prompt}] + messages[-3:]
        chatbot_output = get_chatbot_response(self.client, self.model_name, input_messages)
        chatbot_output = double_check_json_output(self.client, self.model_name, chatbot_output)
        output = self.postprocess(chatbot_output)
        
        return output
    

    def postprocess(self, output):
        #print("Guard_Agent Output:", output)  
        try:
            output = output.strip()
            output_json = json.loads(output)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in chatbot response: {output}. Error: {str(e)}") from e
    
        dict_output = {
            "role": "assistant",
            "content": output_json["message"],
            "memory": {
                "agent": "guard_agent",
                "guard_decision": output_json["decision"],
            },
        }
        return dict_output

