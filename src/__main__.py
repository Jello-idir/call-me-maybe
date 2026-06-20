from llm_sdk import Small_LLM_Model
import numpy as np


llm = Small_LLM_Model()

while True:
    user_input = input("--> ")
    if user_input.lower() == "quit":
        break
    while True:
