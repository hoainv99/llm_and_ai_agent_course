import os
import gradio as gr
from src.rag_system import RAGSystem
from src.pdf_processor import load_and_split_pdf
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
            self.rag = RAGSystem(collection_name=os.getenv("collection_name"),vector_store_type="qdrant")
            self.rag.setup_qa_chain()
        except Exception as e:
            return f"Error init RAG system: {str(e)}"
    
    def respond(self, message, history):
        """Generate response for the chat message"""
        if self.rag is None:
            return "Please upload a PDF file first."
        
        try:
            # Add current message to history
            self.chat_history.append({"role": "user", "content": message})
            
            # Create context from history
            history_context = "\n".join([
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in self.chat_history[-5:]  # Use last 5 messages for context
            ])
            
            # Add history context to the query
            query_with_history = f"""Previous conversation:
{history_context}

Current question: {message}"""
            
            # Get response from RAG system
            result = self.rag.query(query_with_history)
            print(result)
            # Add response to history
            self.chat_history.append({"role": "assistant", "content": result["result"].content})
            
            return result["result"].content
        except Exception as e:
            return f"Error generating response: {str(e)}"

def create_interface():
    chat_interface = ChatInterface()
    
    with gr.Blocks(title="PDF Chat Assistant") as demo:
        gr.Markdown("# PDF Chat Assistant")
        gr.Markdown("Upload a PDF file and ask questions about its content.")
        
        with gr.Row():
            with gr.Column(scale=1):
                pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                upload_button = gr.Button("Process PDF")
                status = gr.Textbox(label="Status", interactive=False)
            
            with gr.Column(scale=2):
                chatbot = gr.ChatInterface(
                    chat_interface.respond,
                    title="Chat with your PDF",
                    description="Ask questions about your PDF content",
                    theme="soft",
                    examples=[
                        "What is the main topic of the document?",
                        "Can you summarize the key points?",
                        "What are the main findings?"
                    ]
                )
        
        upload_button.click(
            chat_interface.initialize_rag,
            inputs=[pdf_input],
            outputs=[status]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True) 