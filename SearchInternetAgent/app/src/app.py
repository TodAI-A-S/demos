from langchain.utilities import GoogleSerperAPIWrapper
from langchain import LLMMathChain, OpenAI
from langchain.agents import Tool
from streamlit_chat import message
import streamlit as st
from langchain.utilities.zapier import ZapierNLAWrapper
from langchain.agents import AgentType
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.agents import initialize_agent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationBufferMemory

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain import SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import re
from getpass import getpass
from datetime import date

import os
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]


llm = OpenAI(model_name='text-davinci-003', temperature=0.4)
search = GoogleSerperAPIWrapper()
llm_math_chain = LLMMathChain(llm=llm, verbose=True)

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when asked to search for answers to questions about current events."
    ),
    Tool(
        func=llm_math_chain.run,
        name="Calculator",
        description="useful for when you need to answer questions about math."
    )
]

sys_msg = f"""You are an assistent helping the company 'Forsikring and Pension'. They need your help preparing for debates. Asides from this follow this: TodBot is a large language model.
    
    TodBot is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, TodBot is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

    TodBot is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, TodBot is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

    Overall, TodBot is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist. Answer the following questions as best you can using your tools. If you cannot find the answer using your tools it is okay to ask for more context. The current date is {date.today()}. The current year is {date.today().year}.
"""


if "memory" not in st.session_state:
    st.session_state['memory'] = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )


agent = initialize_agent(
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    verbose=True,
    memory=st.session_state['memory'],
    agent_kwargs={

        "system_message": sys_msg

    }
)

st.header("AI Agent")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []


def get_text():
    input_text = st.text_input(" ", "Hej, hvem er du?", key="input")
    return input_text


user_input = get_text()
output = agent.run(user_input)
st.session_state.past.append(user_input)
st.session_state.generated.append(output)

if st.session_state['generated']:

    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state['generated'][i],
                key=str(i), avatar_style='bottts', seed=5)
        message(st.session_state['past'][i], is_user=True,
                avatar_style='thumbs', seed=39, key=str(i) + '_user')
