from langchain.chains.combine_documents import create_stuff_documents_chain          # 단일 체인 결합 라이브러리
from langchain.chains import create_history_aware_retriever, create_retrieval_chain  # 채팅 기록을 위한 라이브러리
from langchain_core.output_parsers import StrOutputParser                            # 문자열 출력 파서 라이브러리
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder           # 프롬프트 템플릿 라이브러리
from langchain_openai import ChatOpenAI, OpenAIEmbeddings                            # OpenAI 라이브러리
from langchain_pinecone import PineconeVectorStore                                   # Pinecone 벡터DB 라이브러리

from langchain_community.chat_message_histories import ChatMessageHistory            # 챗봇 대화 기록관리 라이브러리
from langchain_core.chat_history import BaseChatMessageHistory                       # 대화기록 관리를 위한 추상 기반 클래스
from langchain_core.runnables.history import RunnableWithMessageHistory              # 객체에 대한 히스토리 관리 라이브러리

store = {}

# 챗봇 히스토리 세션 함수
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# Retriever 함수
def get_retriever():
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')  # 신규 모델을 사용하기 위해 추가
    index_name = 'tax-markdown-index' # RDB 에서 테이블명과 같다고 보면됨.
    database = PineconeVectorStore.from_existing_index(embedding=embeddings, index_name=index_name)
    retriever = database.as_retriever(search_kwargs={'k': 4})
    return retriever


# LLM 가져오는 함수
def get_llm(model='gpt-4o'):
    return ChatOpenAI(model=model)


# 질문전 질의를 LLM을 통해 필터링 하는 함수
def get_dictionary_chain() :
    dictionary = ['사람을 나타내는 표현 -> 거주자']
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        사전: {dictionary}

        질문: {{question}}
    """)

    dictionary_chain = prompt | llm | StrOutputParser()  # 질문을 먼저 LLM 에 태워서 필터(사람을 직장인으로)

    return dictionary_chain


# QAChain 가져오는 함수
def get_rag_chain():
    llm = get_llm() # LLM 모델
    retriever = get_retriever() # Retriever 모델

    # prompt = hub.pull("rlm/rag-prompt") # LangChain 에서 괜찮은 Prompt들을 모아둔 곳.

    # 사용자 질문이 이전 대화 히스토리에 의존할 경우, 독립적인 질문으로 변환하는 프롬프트
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    # 시스템 메시지와 사용자 입력을 기반으로 대화 히스토리를 포함한 프롬프트 템플릿 생성
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # 대화 히스토리를 고려하여 질문을 재구성한 뒤, Retriever 로 관련 문서 검색
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 검색된 문서를 사용하여 질문에 간결하고 정확한 답변을 생성하는 프롬프트
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    # 위 프롬프트를 통해 답변 생성 프롬프트 템플릿 생성
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # 검색된 문서를 기반으로 LLM이 답변을 생성하도록 체인 구성
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 검색체인(history_aware_retriever) 과 QA 체인(question_answer_chain)을 결합해서 검색 및 생성 작업이 하나의 체인으로 실행
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # 최종 체인(rag_chain)에 대화 히스토리 관리 기능 추가
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,                           # 대화 히스토리 관리 기능 추가
        get_session_history,                 # 세션별 대화 히스토리 관리(chat_history 기반)
        input_messages_key="input",          # 사용자 입력 메시지의 키
        history_messages_key="chat_history", # 대화 히스토리를 저장 및 참조하는 키
        output_messages_key="answer",        # 최종 생성된 답변이 저장될 키
    ).pick('answer') # 체인의 출력에서 answer 만 반환

    return conversational_rag_chain

# LLM 질의 함수
def get_ai_response(user_message):
    dictionary_chain = get_dictionary_chain()
    qa_chain = get_rag_chain()
    tax_chain = {"input": dictionary_chain} | qa_chain # 사용자 질문을 필터링 후 QAChain 으로 전달

    # 체인 실행
    ai_response = tax_chain.stream( # invoke 가 아닌 stream 으로 바꾸면 한글자씩 출력된다.
        {
            "question": user_message
        },
        config={
            "configurable":{"session_id":"abc123"}
        }
    )

    return ai_response