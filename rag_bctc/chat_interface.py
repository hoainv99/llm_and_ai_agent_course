import os
import gradio as gr
from rag_system import RAGSystem
from pdf_processor import load_and_split_pdf

class ChatInterface:
    def __init__(self):
        self.rag = None
        self.chat_history = []
        
    def initialize_rag(self, pdf_file):
        """Initialize RAG system with the uploaded PDF file"""
        if pdf_file is None:
            return "Please upload a PDF file first."
        
        try:
            # Save the uploaded file temporarily
            temp_path = "data/hpg.pdf"
            os.makedirs("data", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(pdf_file.read())
            
            # Initialize RAG system
            self.rag = RAGSystem()
            documents = load_and_split_pdf(temp_path)
            print("documents", documents)
            self.rag.create_vector_store(documents)
            self.rag.setup_qa_chain()
            
            # Clean up
            os.remove(temp_path)
            return "PDF processed successfully! You can now start chatting."
        except Exception as e:
            return f"Error processing PDF: {str(e)}"
    
    def respond(self, message, history):
        """Generate response for the chat message"""
        if self.rag is None:
            return "Please upload a PDF file first."
        
        try:
            result = self.rag.query(message)
            return result['result']
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