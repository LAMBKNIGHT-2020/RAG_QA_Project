import os
import traceback
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    from langchain_community.embeddings import OllamaEmbeddings

llm = Ollama(
    base_url="http://10.4.77.115:11434",
    model="qwen3:14b"
)

def simple_chat():
    print("开始聊天,输入quit退出\n")
    while True:
        user_input = input("用户输入: ")
        if user_input.lower() == "quit":
            break
        
        prompt = ChatPromptTemplate.from_messages([("human", "{user_input}")])
        chain = prompt | llm
        response = chain.invoke({"user_input": user_input})
        print(f"AI回答:{response}")

def process_pdf():
    
    pdf_path = input("请输入PDF文件完整路径(例: K:/BS/file.pdf): ")
    
    # 标准化路径格式
    pdf_path = os.path.normpath(pdf_path)
    print(f"尝试加载文件: {pdf_path}")

    try:
        # 检查文件是否存在
        if not os.path.exists(pdf_path):
            print(f"错误: 文件不存在 - {pdf_path}")
            return
            
        # 检查文件是否可读
        try:
            with open(pdf_path, 'rb') as f:
                pass
        except IOError:
            print(f"错误: 无法读取文件 - {pdf_path}")
            return
            
        print("文件检查通过，开始加载...")
        loader = PyMuPDFLoader(pdf_path)  # 修改这里
        documents = loader.load()
        print(f"PDF加载成功,共{len(documents)}页")

        # 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # 单个文本块大小
            chunk_overlap=200  # 文本块之间
        )
        chunks = text_splitter.split_documents(documents)
        print(f"文档被分割成{len(chunks)}个文本块")

        # 初始化嵌入模型和向量数据库
        print("初始化嵌入模型...")
        embeddings = OllamaEmbeddings(
            base_url="http://10.4.77.115:11434",
            model="qwen3:14b"
        )
        
        # 把文本块转换成向量存入向量库
        print("创建向量存储...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./vector_db"
        )

        #vectorstore.persist()  # 持久化保存
        print("文本块已成功向量化并存入数据库")

        print("\n-------现在可以基于文档提问-------")

        while True:
            user_q = input("\n你的问题(输入'back'返回主菜单): ")
            if user_q.lower() == "back":
                break

            # 检索与生成
            # 检索出最相关的文本块
            relevant_chunks = vectorstore.similarity_search(user_q, k=3)
            # 文本块内容合并为上下文字符串
            context_text = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "请严格根据以下上下文来回答问题，如果上下文中没有答案，就说你不知道。\n上下文:{context}"),
                ("human", "{question}")
            ])
            chain = prompt_template | llm

            # 调用模型
            response = chain.invoke({
                "context": context_text,
                "question": user_q
            })
            print(f"\nAI:{response}")

    except Exception as e:
        print(f"加载时发生错误: {e}")
        print("详细错误信息:")
        traceback.print_exc()

def main():
    while True:
        print("\n" + "=" * 50)
        print("RAG系统菜单")
        print("1. 开始简单对话")
        print("2. 处理PDF文档")
        print("3. 退出系统")
        print("=" * 50)

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