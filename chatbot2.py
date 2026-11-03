import gradio as gr
from getpass import getpass

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Step 1: Get Groq API Key
api_key = getpass("Enter your Groq API Key: ")

# Step 2: Initialize the LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=api_key,
    temperature=0.7
)

# Step 3: Create a Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and friendly AI assistant. Answer clearly."),
    ("human", "{question}")
])

# Step 4: Create the LangChain
chain = prompt | llm

# Step 5: Define Chatbot Function
def chatbot_response(message, history):
    response = chain.invoke({"question": message})
    return response.content

# Step 6: Create Chatbot UI
demo = gr.ChatInterface(
    fn=chatbot_response,
    title="Simple AI Chatbot",
    description="Ask any question and get an AI-generated answer.",
    textbox=gr.Textbox(
        placeholder="Type your question here...",
        container=True
    )
)

# Step 7: Launch the Application
demo.launch()