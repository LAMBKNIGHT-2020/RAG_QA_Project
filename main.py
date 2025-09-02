import os
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader


llm = Ollama(
    base_url = "http://10.4.77.115:11434",
    model = "qwen3:14b"
)

def simple_chat(user_input):
    print("开始聊天,输入quit退出\n")
    print(f"用户输入: {user_input}")
    while True:
            user_input = input("用户输入: ")
            if user_input.lower() == "quit":
                  break
            
            prompt = ChatPromptTemplate.from_messages([("human","{user_input}")])
            chain = prompt | llm
            response = chain.invoke({"user_input": user_input})
            print("AI回答:{response}")
