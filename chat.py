from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

def get_llm_and_agent() -> AgentExecutor:
    """
    Khởi tạo Language Model và Agent với cấu hình đơn giản
    Returns:
        AgentExecutor: Agent đã được cấu hình với:
            - Model: GPT-4
            - Temperature: 0
            - Streaming: Enabled
            - Custom system prompt
    Chú ý:
        - Yêu cầu OPENAI_API_KEY đã được cấu hình
        - Agent được thiết lập với tên "ChatchatAI"
        - Sử dụng chat history để duy trì ngữ cảnh hội thoại
    """
    # Khởi tạo ChatOpenAI với chế độ streaming
    llm = ChatOpenAI(temperature=0, streaming=True, model="gpt-4")
    
    # Thiết lập prompt template cho agent
    system = """You are an expert at AI. Your name is ChatchatAI."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Tạo và trả về agent
    agent = create_openai_functions_agent(llm=llm, tools=[], prompt=prompt)
    return AgentExecutor(agent=agent, tools=[], verbose=True)

# Khởi tạo agent
agent_executor = get_llm_and_agent()