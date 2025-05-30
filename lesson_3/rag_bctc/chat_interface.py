import os
import gradio as gr
from src.rag_qdrant import RAGSystem
from dotenv import load_dotenv

load_dotenv()
class ChatInterface:
    def __init__(self):
        self.rag = None
        self.chat_history = []
        self.initialize_rag()
        
    def initialize_rag(self):
        """"""
        try:
            print("collection:", os.getenv("collection_name"))
            self.rag = RAGSystem(collection_name=os.getenv("collection_name"))
            self.rag.setup_qa_chain()
            print("init RAG success")
        except Exception as e:
            return f"Error init RAG system: {str(e)}"
    
    def respond(self, message, history=None):
        """Generate response for the chat message"""
        if self.rag is None:
            return "Please upload a PDF file first."
        
        try:
            # Get response from RAG system
            result = self.rag.query(message)

            return result["result"].content
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def clear_history(self):
        """Clear the chat and RAG system history"""
        if self.rag is not None:
            self.rag.clear_history()

def create_interface():
    chat_interface = ChatInterface()
    
    with gr.Blocks(title="PDF Chat Assistant") as demo:
        gr.Markdown("# PDF Chat Assistant")
        gr.Markdown("Upload a PDF file and ask questions about its content.")
        
        with gr.Row():
            with gr.Column():
                chatbot = gr.ChatInterface(
                    fn=chat_interface.respond,
                    title="Chat with your PDF",
                    description="Ask questions about your PDF content",
                    theme="soft",
                    examples=[
                        "Tài liệu này nói về điều gì?",
                        "Hãy liệt kê các nội dung chính trong tài liệu",
                    ]
                )
                clear_btn = gr.Button("Clear History")
                def clear_all():
                    chat_interface.clear_history()
                    return None
                clear_btn.click(
                    clear_all,
                    outputs=[chatbot],
                )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True) 