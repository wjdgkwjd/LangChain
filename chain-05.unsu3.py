from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv 

load_dotenv() # openai_key  .env 선언 사용 

index= FAISS.load_local("faiss-unsu-txt", OpenAIEmbeddings())

llm_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)  

chain = load_qa_chain(llm_model, chain_type="stuff")


query = "아내가 좋아하는 음식은?"
docs = index.similarity_search(query)
res = chain.run(input_documents=docs, question=query)
print( query , res)

query = "주인공의 직업은?"
docs = index.similarity_search(query)
res = chain.run(input_documents=docs, question=query)
print( query ,res)

query = "지은이?"
docs = index.similarity_search(query)
res = chain.run(input_documents=docs, question=query)
print( query , res)