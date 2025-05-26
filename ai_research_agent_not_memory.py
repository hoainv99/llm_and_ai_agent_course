"""
AI Research Agent using LangChain

This agent helps research AI topics by:
1. Collecting information about AI topics using search tools
2. Organizing and summarizing the information
3. Providing insights and analysis about AI trends and developments
"""

import os
from typing import List, Dict, Any, Optional
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.tools import tool, DuckDuckGoSearchRun
from langchain.chains import LLMChain
from langchain.schema import SystemMessage, HumanMessage
from langchain.agents import initialize_agent, create_structured_chat_agent
from langchain import hub

import gradio as gr

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
GOOGLE_API_KEY = "AIzaSyDEKPwEcRb6bRxZjc0iybC0YokwLct3FQE"

class AIResearchAgent:
    def __init__(self, api_key: str):
        """Initialize the AI Research Agent."""
        self.api_key = api_key
        self.llm = self._initialize_llm()
        self.search_tool = DuckDuckGoSearchRun()
        self.tools = self._create_tools()
        self.agent = self._create_agent()
    
    def _initialize_llm(self):
        """Initialize the language model."""
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=self.api_key,
            temperature=0.7,
            convert_system_message_to_human=True
        )
    
    def _create_tools(self):
        """Create and return the tools for the agent."""
        
        @tool
        def search_ai_topics(query: str) -> str:
            """Search for information about AI topics. Use this tool for finding information about artificial intelligence,
            machine learning, deep learning, NLP, and related topics."""
            try:
                # Improve search by adding AI domain context
                enhanced_query = f"AI technology {query} latest research developments"
                results = self.search_tool.run(enhanced_query)
                return results[:2000]  # Limit result length
            except Exception as e:
                return f"Error searching for '{query}': {str(e)}"

        @tool
        def analyze_ai_trend(trend_name: str) -> str:
            """Analyze a specific AI trend or technology and provide insights about its development, applications, and future prospects."""
            try:
                # Search for the trend and recent developments
                search_query = f"{trend_name} AI technology latest developments impact applications future"
                search_results = self.search_tool.run(search_query)
                
                # Use the LLM to analyze the results
                analysis_prompt = f"""
                Based on the following information about {trend_name} in AI, provide a detailed analysis covering:
                1. Key developments and current state
                2. Main applications and use cases
                3. Future prospects and potential impact
                4. Challenges and limitations
                
                Information:
                {search_results[:1500]}
                
                Provide a well-structured, insightful analysis with concrete examples and facts.
                """
                
                analysis = self.llm.invoke([HumanMessage(content=analysis_prompt)])
                return analysis.content
            except Exception as e:
                return f"Error analyzing AI trend '{trend_name}': {str(e)}"

        @tool
        def compare_ai_technologies(tech1: str, tech2: str) -> str:
            """Compare two AI technologies, approaches, or models. Highlight differences, strengths, weaknesses, and suitable applications."""
            try:
                # Search for comparative information
                search_query = f"compare {tech1} vs {tech2} AI technology differences strengths weaknesses applications"
                search_results = self.search_tool.run(search_query)
                
                # Use the LLM to create a structured comparison
                comparison_prompt = f"""
                Based on the following information, create a detailed comparison between {tech1} and {tech2} in AI:
                
                Information:
                {search_results[:1500]}
                
                Your comparison should include:
                1. Brief overview of each technology
                2. Key technical differences
                3. Strengths and weaknesses of each
                4. Suitable applications and use cases
                5. Future outlook
                
                Present this as a structured, balanced comparison with specific examples where possible.
                """
                
                comparison = self.llm.invoke([HumanMessage(content=comparison_prompt)])
                return comparison.content
            except Exception as e:
                return f"Error comparing AI technologies '{tech1}' and '{tech2}': {str(e)}"

        @tool
        def general_calculation(expression: str) -> str:
            """Perform mathematical calculations. Can handle basic arithmetic and also work with previous results from conversation.
            Examples: '7 * 8', '56 * 2', '112 / 4', etc."""
            try:
                # Safety check - only allow basic mathematical operations
                allowed_chars = set('0123456789+-*/().= ')
                if not all(c in allowed_chars for c in expression.replace(' ', '')):
                    return "Error: Only basic mathematical operations are allowed."
                
                # Evaluate the expression safely
                result = eval(expression)
                return f"The result of {expression} is {result}"
            except Exception as e:
                return f"Error calculating '{expression}': {str(e)}"
        
        return [search_ai_topics, analyze_ai_trend, compare_ai_technologies, general_calculation]
    
    def _create_agent(self):
        """Create and return the conversational agent."""
        # Create a proper conversational agent that uses memory
        try:
            # Use a conversational prompt that properly incorporates chat history
            template = """You are an AI Research Assistant specialized in artificial intelligence and machine learning domains.
You can also perform mathematical calculations and remember previous conversation context.

TOOLS:
{tools}

IMPORTANT: Always check the chat history below for context from previous questions and answers.
Pay special attention to previous calculation results that the user might reference.

CHAT HISTORY:
{chat_history}

Use the following format for your responses:

Question: the input question you must answer
Thought: consider the chat history and think about what to do (check if user refers to previous results)
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer (consider the full context)
Final Answer: the final answer to the original input question

Guidelines:
1. ALWAYS check chat history for context and previous results
2. If user refers to "kết quả trên", "result above", or previous calculations, use those values
3. For math: use general_calculation tool
4. For AI search: use search_ai_topics tool  
5. For AI analysis: use analyze_ai_trend tool
6. For AI comparison: use compare_ai_technologies tool

Question: {input}
Thought: {agent_scratchpad}"""

            prompt = PromptTemplate(
                input_variables=["input", "chat_history", "agent_scratchpad", "tools", "tool_names"],
                template=template
            )
            
            agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=prompt
            )
            
        except Exception as e:
            logger.warning(f"Failed to create react agent: {e}")
            # Fallback to structured chat agent
            template = """You are an AI Research Assistant. You can perform calculations and research AI topics.

                You have access to the following tools:
                {tools}

                Use a json blob to specify a tool:
                ```
                {{
                "action": $TOOL_NAME,
                "action_input": $INPUT
                }}
                ```

                Valid actions: "Final Answer" or {tool_names}

                Question: {input}
                Thought: {agent_scratchpad}"""

            prompt = PromptTemplate(
                input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
                template=template
            )
            
            agent = create_structured_chat_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=prompt
            )
        
        # Create agent executor with memory properly integrated
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors="Check your output and make sure it conforms! Look at chat history for context.",
            max_iterations=5,
            return_intermediate_steps=False
        )
        
        return agent_executor
    
    def query(self, question: str) -> str:
        """Process a user query and return the response."""
        try:
            # Use the agent with proper memory integration
            response = self.agent.invoke({"input": question})
            return response.get("output", "I'm sorry, I couldn't process your request.")
            
        except Exception as e:
            logger.error(f"Agent error: {e}")
            # If agent fails, provide a simple fallback
            return f"I encountered an error processing your request. Could you please rephrase your question? Error: {str(e)}"
    
    # def clear_memory(self):
    #     """Clear the conversation memory."""
    #     self.memory.clear()

def create_gradio_interface(agent: AIResearchAgent):
    """Create a Gradio interface for the AI Research Agent."""
    
    def chat_function(message, history):
        try:
            response = agent.query(message)
            return response
        except Exception as e:
            return f"Error: {str(e)}"
    
    # def clear_history():
    #     agent.clear_memory()
    #     return []
    
    # Create the Gradio interface
    with gr.Blocks(title="AI Research Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 AI Research Agent")
        gr.Markdown("Ask questions about AI technologies, trends, and developments. I can also help with basic calculations!")
        
        chatbot = gr.Chatbot(
            height=500,
            show_label=False,
            container=True,
            bubble_full_width=False
        )
        
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ask me about AI topics or calculations...",
                container=False,
                scale=7
            )
            submit_btn = gr.Button("Send", scale=1, variant="primary")
            clear_btn = gr.Button("Clear", scale=1)
        
        # Examples
        gr.Examples(
            examples=[
                "What are the latest developments in large language models?",
                "Compare GPT-4 and Claude in terms of capabilities",
                "Analyze the trend of AI in healthcare",
                "Calculate 7 * 8",
                "What is the future of autonomous vehicles?",
                "Explain transformer architecture in simple terms"
            ],
            inputs=msg
        )
        
        def respond(message, chat_history):
            if message.strip():
                bot_message = chat_function(message, chat_history)
                chat_history.append((message, bot_message))
            return "", chat_history
        
        # def clear_conversation():
        #     clear_history()
        #     return []
        
        # Set up event handlers
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
        # clear_btn.click(clear_conversation, outputs=[chatbot])
    
    return demo

def main():
    """Main function to run the AI Research Agent."""
    try:
        # Initialize the agent
        agent = AIResearchAgent(GOOGLE_API_KEY)
        
        # Test the agent with the original question
        print("Testing the agent...")
        test_response = agent.query("Tính giúp tôi 7 * 8")
        print(f"Test Response: {test_response}")
        
        # Create and launch the Gradio interface
        demo = create_gradio_interface(agent)
        demo.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860
        )
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print(f"Failed to start the application: {e}")

if __name__ == "__main__":
    main()