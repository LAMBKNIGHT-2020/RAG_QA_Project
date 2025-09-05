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

def process_pdf():
    pdf_path = input("请输入PDF文件完整路径（例./K:/BS/file.pdf）: ")

    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
            
        print(f"\n 成功加载PDF文档")
        print(f"\n 文档共{len(documents)}个页面")
        print(f"\n 第一页内容预览：\n")
        print(documents[0].page_content[0:500] + "...")

        choice = input("\n 请选择操作：1.基于文档提问 2.返回  ")
        if choice == "1":
            print("开发中")
        
    except Exception as e:
        print(f"加载时发生错误: {e}")

def main():
    while True:
        print("\n" + "="*50)
        print("RAG系统菜单")
        print("1. 开始简单对话")
        print("2. 处理PDF文档")
        print("3. 退出系统")
        print("="*50)

        choice = input("请选择功能 [1/2/3]: ")
        
        if choice == '1':
            simple_chat()
        elif choice == '2':
            process_pdf()
        elif choice == '3':
            print("再见！")
            break
        else:
            print("无效选择，请重新输入。")

if __name__ == "__main__":
    main()