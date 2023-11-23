from dotenv import load_dotenv;load_dotenv() # openai_key  .env 선언 사용 
#@title 5. Agents and Tools

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
import wikipedia
import  math

chat = ChatOpenAI(model_name='gpt-4', temperature=0.9)
tools = load_tools(["wikipedia", "llm-math"], llm=chat)
# tools = load_tools(["serpapi", "llm-math"], llm=chat)

agent = initialize_agent(tools, llm=chat, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)
query = "대한민국 대통령 이름과 퇴임때 나이, 현재는 2023년 10월 21일. 한국어로 답변"
res=agent.run(query )
print(  res  )