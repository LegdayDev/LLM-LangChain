from langchain_core.output_parsers import StrOutputParser # 문자열 출력 파서 라이브러리
from langchain_core.prompts import ChatPromptTemplate     # 프롬프트 템플릿 라이브러리
from langchain.chains import RetrievalQA                  # RetrievalQA 체인 라이브러리
from langchain import hub                                 # LangChain Hub 라이브러리
from langchain_openai import ChatOpenAI                   # OpenAI의 GPT모델로 대화형AI 라이브러리
from langchain_openai import OpenAIEmbeddings             # 임베딩 모델 라이브러리
from langchain_pinecone import PineconeVectorStore        # Pinecone 벡터DB 라이브러리


# LLM 질의 함수
def get_ai_message(user_message):
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')  # 기본모델은 002인 예전모델이다. 신규 모델을 사용하기 위해 추가
    index_name = 'tax-markdown-index'
    database = PineconeVectorStore.from_existing_index(embedding=embeddings, index_name=index_name)
    llm = ChatOpenAI(model='gpt-4o')
    prompt = hub.pull("rlm/rag-prompt")
    retriever = database.as_retriever(search_kwargs={'k': 4})
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    dictionary = ['사람을 나타내는 표현 -> 거주자']

    prompt = ChatPromptTemplate.from_template(
    f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        사전: {dictionary}

        질문: {{question}}
    """)

    dictionary_chain = prompt | llm | StrOutputParser()  # StrOutputParser() 는 결과를 String으로
    tax_chain = {"query": dictionary_chain} | qa_chain  # 파싱
    ai_message = tax_chain.invoke({"question": user_message})
    return ai_message["result"]