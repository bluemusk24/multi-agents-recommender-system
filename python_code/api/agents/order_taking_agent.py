import os
import json
from .utils import get_chatbot_response,double_check_json_output
from openai import OpenAI
from copy import deepcopy
from dotenv import load_dotenv, find_dotenv

# initialize .env file
_ = load_dotenv(find_dotenv())


class OrderTakingAgent():
    def __init__(self, recommendation_agent):
        self.client = OpenAI(
            api_key=os.getenv("RUNPOD_TOKEN"),
            base_url=os.getenv("RUNPOD_CHATBOT_URL"),
        )
        self.model_name = os.getenv("MODEL_NAME")

        self.recommendation_agent = recommendation_agent
    
    def get_response(self,messages):
        messages = deepcopy(messages)
        system_prompt = """
        You are a customer support Bot for a coffee shop called "Merry's way".

        Here is the menu for this coffee shop:

        Cappuccino - $4.50
        Jumbo Savory Scone - $3.25
        Latte - $4.75
        Chocolate Chip Biscotti - $2.50
        Espresso shot - $2.00
        Hazelnut Biscotti - $2.75
        Chocolate Croissant - $3.75
        Dark chocolate (Drinking Chocolate) - $5.00
        Cranberry Scone - $3.50
        Croissant - $3.25
        Almond Croissant - $4.00
        Ginger Biscotti - $2.50
        Oatmeal Scone - $3.25
        Ginger Scone - $3.50
        Chocolate syrup - $1.50
        Hazelnut syrup - $1.50
        Carmel syrup - $1.50
        Sugar Free Vanilla syrup - $1.50
        Dark chocolate (Packaged Chocolate) - $3.00

        Things to **NOT** do:
        * **DON'T** ask how to pay by cash or card. You should **never** ask or talk about payment methods.
        * Don't tell the user to go to the counter.
        * Don't tell the user to go to a place to get the order.
        * Avoid asking any unnecessary questions that are irrelevant to the order process.

        Your task is as follows:
        1. Take the User's Order.
        2. Validate that all their items are on the menu.
        3. If an item is not on the menu, inform the user and repeat back the valid order.
        4. Ask if they need anything else.
        5. If they don't need anything else, close the conversation with the following:
            - List all the items and their prices.
            - Calculate the total.
            - Thank the user for the order and close the conversation with no more questions.
        6. If the user has already been recommended items, do not ask again and proceed directly to completing the order.

        The user message will contain a section called "memory". This section will contain:
        "order"
        "step number"
        Please utilize this information to determine the next step in the process.
                
        Produce the following output without any additions, not a single letter outside the structure below. Your output should strictly follow the format:
        {
        "chain of thought": "Write down your critical thinking about what is the maximum task number the user is on right now. Then write down your critical thinking about the user input and its relation to the coffee shop process. Then write down your thinking about how you should respond in the response parameter while considering the 'Things to NOT DO' section. Focus on the things that you should not do.",
        "step number": "Determine which task you are on based on the conversation.",
        "order": [{"item": "item_name", "quantity": "quantity", "price": "price"}],
        "response": "Write a response to the user"
        }
        Ensure to output just the order from the output and nothing else.
        """

        # get last message (which will be the order) from the apriori engine
        last_order_taking_status = ""
        asked_recommendation_before = False
        
        for message_index in range(len(messages)-1,0,-1):
            message = messages[message_index]
            
            agent_name = message.get("memory",{}).get("agent","")
            if message["role"] == "assistant" and agent_name == "order_taking_agent":
                step_number = message["memory"]["step number"]
                order = message["memory"]["order"]
                asked_recommendation_before = message["memory"]["asked_recommendation_before"]
                last_order_taking_status = f"""
                step number: {step_number}
                order: {order}
                """
                break

        messages[-1]["content"] = last_order_taking_status + " \n "+ messages[-1]["content"]
        input_messages = [{"role": "system", "content": system_prompt}] + messages        
        chatbot_output = get_chatbot_response(self.client,self.model_name,input_messages)

        # double check json 
        chatbot_output = double_check_json_output(self.client,self.model_name,chatbot_output)
        output = self.postprocess(chatbot_output,messages,asked_recommendation_before)
        return output
    

    def postprocess(self,output,messages,asked_recommendation_before):

        try:
            #output = output.strip()
            output = json.loads(output)

            if type(output["order"]) == str:
                output["order"] = json.loads(output["order"])

            response = output["response"]
            if not asked_recommendation_before and len(output["order"])>0:
                recommendation_output = self.recommendation_agent.get_recommendations_from_order(messages,output["order"])
                response = recommendation_output["content"]
                asked_recommendation_before = True
        except json.decoder.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in chatbot response: {output}. Error: {str(e)}") from e

        dict_output = {
            "role": "assistant",
            "content": response ,
            "memory": {"agent":"order_taking_agent",
                       "step number": output["step number"],
                       "order": output["order"],
                       "asked_recommendation_before": asked_recommendation_before
                      }
        }

        return dict_output
