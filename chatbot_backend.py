# chatbot_backend.py

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List, Dict, Optional
import os

class ChatbotBackend:
    def __init__(self, api_key: Optional[str] = None):
        """
        Khởi tạo backend của chatbot
        Args:
            api_key: OpenAI API key. Nếu không cung cấp sẽ lấy từ environment variable
        """
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            
        self.llm = ChatOpenAI(temperature=0, streaming=True, model="gpt-4")
        self.agent_executor = self._create_agent()
        self.chat_history: List[Dict] = []

    def _create_agent(self) -> AgentExecutor:
        """
        Tạo agent với cấu hình cơ bản
        """
        system = """You are an expert at AI. Your name is ChatchatAI."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_openai_functions_agent(llm=self.llm, tools=[], prompt=prompt)
        return AgentExecutor(agent=agent, tools=[], verbose=True)

    async def process_message(self, message: str, callback=None) -> str:
        """
        Xử lý tin nhắn từ user và trả về response
        Args:
            message: Tin nhắn từ user
            callback: Function để handle streaming response (optional)
        Returns:
            str: Response từ chatbot
        """
        # Chuẩn bị input với chat history
        input_data = {
            "input": message,
            "chat_history": self.chat_history
        }

        try:
            # Xử lý message và lấy response
            response = await self.agent_executor.ainvoke(
                input_data,
                callbacks=[callback] if callback else None
            )
            
            # Cập nhật chat history
            self.chat_history.append({"role": "human", "content": message})
            self.chat_history.append({"role": "assistant", "content": response["output"]})
            
            return response["output"]
            
        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            print(error_msg)
            return error_msg

    def clear_history(self):
        """Xóa lịch sử chat"""
        self.chat_history = []

    def get_chat_history(self) -> List[Dict]:
        """Lấy lịch sử chat"""
        return self.chat_history