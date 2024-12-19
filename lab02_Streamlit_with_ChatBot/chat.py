import streamlit as st # Streamlit 라이브러리

from langchain_core.output_parsers import StrOutputParser # 문자열 출력 파서 라이브러리
from langchain_core.prompts import ChatPromptTemplate     # 프롬프트 템플릿 라이브러리
from langchain.chains import RetrievalQA                  # RetrievalQA 체인 라이브러리
from langchain import hub                                 # LangChain Hub 라이브러리
from langchain_openai import ChatOpenAI                   # OpenAI의 GPT모델로 대화형AI 라이브러리
from dotenv import load_dotenv                            # 환경변수 라이브러리
from langchain_openai import OpenAIEmbeddings             # 임베딩 모델 라이브러리
from langchain_pinecone import PineconeVectorStore        # Pinecone 벡터DB 라이브러리

load_dotenv()  # 환경변수 불러오기

st.set_page_config(page_title='소득세 챗봇', page_icon="😶‍🌫️")

st.title("😒소득세 챗봇")
st.caption("소득세에 관련된 모든것을 답해드립니다!")

# st.session_state 는 Streamlit 의 세션 상태 관리 객체, Key-Value 구조.
if 'message_list' not in st.session_state:
    st.session_state.message_list = []


# 세션에서 저장된 메시지 화면에 출력.
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

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

#:=(윌러스)연산자는 Python 3.8에 도입된 기능이다.
#구조 -> variable := expression
#해석 -> expression 의 값을 평가한 뒤, 그 결과를 variable에 저장.
if user_question := st.chat_input(placeholder='소득세에 관련된 궁금한 내용들을 말씀해주세요!'):
    # 사용자
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role":"user", "content":user_question})

    with st.spinner("답변을 생성중입니다..."):
        # 챗봇(AI)
        ai_mesesage = get_ai_message(user_question)
        with st.chat_message("ai"):
            st.write(ai_mesesage)
        st.session_state.message_list.append({"role":"ai", "content":ai_mesesage})


