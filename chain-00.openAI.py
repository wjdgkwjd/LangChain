
from dotenv import load_dotenv;load_dotenv() # openai_key  .env 선언 사용 
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage,  SystemMessage


llm_model= ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.9)

query = "안녕!"
print( query)
res=llm_model.predict(  query )

print(res)


sys = SystemMessage(content="""
                    너의 이름은 제돌이 이다.
                    제주과학고등학교 1학년에 재학 중 이다.
                    나이는 16세이다. 물음에 간결하게 대답한다.
                    교장은 이창훈
                    """)

query = "어느 학교 다녀?"
print( query)
user= HumanMessage(content= query)
res= llm_model([sys, user])
print(res.content)
