from agents import (GuardAgent,
                    ClassificationAgent,
                    DetailsAgent,
                    OrderTakingAgent,
                    RecommendationAgent,
                    AgentProtocol
                    )

from typing import Dict
import os
import pathlib
#import sys
folder_path = pathlib.Path(__file__).parent.resolve()      # to resolve path issues

def main():
    guard_agent = GuardAgent()
    classification_agent = ClassificationAgent()
    recommendation_agent = RecommendationAgent(
                            os.path.join(folder_path, "recommendation_objects/apriori_recommendations.json"),
                            os.path.join(folder_path, "recommendation_objects/popular_recommendation.csv")
                            )
                            
                                    
    agent_dict: Dict[str, AgentProtocol] = {
        "details_agent": DetailsAgent(),
        "recommendation_agent": recommendation_agent,
        "order_taking_agent": OrderTakingAgent(recommendation_agent)
    }
    
    messages = []
    while True:
        # Display the chat history
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("\n\nPrint Messages ...............")
        for message in messages:
            print(f"{message['role']}: {message['content']}")

        # Get user input from terminal
        prompt = input("User: ")
        messages.append({"role": "user", "content": prompt})

        # Get GuardAgent's response from agent_protocol.py
        guard_agent_response = guard_agent.get_response(messages)
        if guard_agent_response["memory"]["guard_decision"] == "not allowed":   # it should print sorry...., but if allowed, continue to classification agent
            messages.append(guard_agent_response)
            continue
        
        # Get ClassificationAgent's response
        classification_agent_response = classification_agent.get_response(messages)
        chosen_agent=classification_agent_response["memory"]["classification_decision"]
    #    print("Chosen Agent: ", chosen_agent)



        # Get the chosen agent's response
        agent = agent_dict[chosen_agent]
        response = agent.get_response(messages)
        
        messages.append(response)

        
if __name__ == "__main__":
    main()