import streamlit as st

st.set_page_config(page_title='소득세 챗봇', page_icon="😶‍🌫️")

st.title("😒소득세 챗봇")
st.caption("소득세에 관련된 모든것을 답해드립니다!")

# st.session_state 는 Streamlit 의 세션 상태 관리 객체, Key-Value 구조.
if 'message_list' not in st.session_state:
    st.session_state.message_list = []


#:=(윌러스)연산자는 Python 3.8에 도입된 기능이다.
#구조 -> variable := expression
#해석 -> expression 의 값을 평가한 뒤, 그 결과를 variable에 저장.

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_question := st.chat_input(placeholder='소득세에 관련된 궁금한 내용들을 말씀해주세요!'):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role":"user", "content":user_question})
