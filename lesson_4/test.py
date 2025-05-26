# import google.generativeai as genai

# genai.configure(api_key="AIzaSyDEKPwEcRb6bRxZjc0iybC0YokwLct3FQE")

# model = genai.GenerativeModel('gemini-2.0-flash')
# response = model.generate_content("Tell me a joke.")
# print(response.text)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # hoặc "gemini-1.5-pro", không có "gemini-2.0-flash" trong LangChain (tính đến nay)
    google_api_key="AIzaSyDEKPwEcRb6bRxZjc0iybC0YokwLct3FQE",
    temperature=0.7,
)

response = llm([HumanMessage(content="Tell me a joke.")])
print(response.content)
from langchain.tools import tool

@tool
def calculate(expression: str) -> str:
    """Tính toán biểu thức toán học."""
    try:
        return str(eval(expression))
    except Exception as e:
        return str(e)

tools = [calculate]

memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)
response = agent.run("Tính giúp tôi 7 * 8")
print(response)

response = agent.run("Cộng thêm 20 nữa")
print(response)
# import gradio as gr
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import RetrievalQA
# from langchain.vectorstores import FAISS
# from langchain.embeddings import GoogleGenerativeAIEmbeddings
# from langchain.document_loaders import TextLoader

# # Load FAISS index đã build từ buổi 3
# db = FAISS.load_local("my_faiss_index", GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
# retriever = db.as_retriever()

# # Khởi tạo LLM Gemini
# # llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key="GEMINI_API_KEY")

# # Tạo chuỗi RAG
# rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# # Hàm dùng cho giao diện
# def ask_rag(query):
#     return rag_chain.run(query)

# # Gradio app
# demo = gr.Interface(fn=ask_rag, inputs=gr.Textbox(label="Nhập câu hỏi"), outputs="text", title="Trợ lý AI với RAG")

# if __name__ == "__main__":
#     demo.launch()