from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole
import streamlit as st

# Your task is to get creative adding different features to the bot in addition to RAG
# In this sample solution we see 3 examples of additional features:
# 1. The option for the user to switch between different system prompts
# 2. The option for the user to alter the LLMs temperature setting
# 3. The ability to reset the chat by typing "goodbye"


# Setting up system prompt options:
prompt_options = {
    'basic_context': (
        "You are a Chatbot conversing with a human about consciousness. Base all your answers on the provided context"
        ),
    'Helpful': (
        'YOU ARE NOW IN HELPFUL MODE, be very constructive and provide solutions and simple explanations. '
        'Reiterate the provided context in a digestible and explanatory way and add analogies where helpful. '
        'Answer all questions clearly and succinctly. '
        ),
    'Unhelpful': (
        'YOU ARE NOW IN UNHELPFUL MODE, you are very sassy and use very idiosyncratic and cryptic language. '
        'Respond to any and all questions from the user very skeptical and destructive'
        'Do not answer question constructively, but reiterate the context in a very complicated and convoluted fashion'
        )
}
# Setting up session state to store current system prompt setting
if 'system_prompts' not in st.session_state:
    st.session_state['system_prompts'] = ['basic_context'] #making it a list allow it to have multiple at once


### INITIALIZING AND CACHING CHATBOT COMPONENTS ###

# Function for initializing the LLM
@st.cache_resource #the result will be cached so it only has to rerun when temp changes
def init_llm(temp=0.01):
    # LLM
    return Groq(
    model="openai/gpt-oss-120b",
    max_new_tokens=768,
    temperature=temp,   # Here we add a variable to play arround with temp
    top_p=0.95,
    repetition_penalty=1.03,
    api_key=st.secrets["GROQ_API_KEY"]
    )


# Function for initializing the retriever
@st.cache_resource #the result will be cached so it only has to rerun when num_chunks changes
def init_rag(num_chunks=2):
    # RAG
    embeddings = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=".cache/embedding_model"
    )
    storage_context = StorageContext.from_defaults(persist_dir="./content/vector_index")
    vector_index = load_index_from_storage(storage_context, embed_model=embeddings)
    return vector_index.as_retriever(similarity_top_k=num_chunks)


# Function for initializing the chatbot memory
@st.cache_resource #the result will be cached so it only has to run once
def init_memory():
    return ChatMemoryBuffer.from_defaults()


# Function for initializing the bot with the specific settings
@st.cache_resource #the result is cached so, unless the parameters are altered, it doesn't need to recreate the bot
def init_bot(prefix_messages, temp=0.01, num_chunks=2):

    # This stuff is cached and only reruns if the parameters change
    llm = init_llm(temp) 
    retriever = init_rag(num_chunks)
    memory = init_memory()

    # Takes the user selections in the session state and turns them into proper ChatMessages
    prefix_messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=prompt_options[system_prompt_selection],
        )
        for system_prompt_selection in prefix_messages
    ]

    # Return initialized bot
    return ContextChatEngine(
        llm=llm, retriever=retriever, memory=memory, prefix_messages=prefix_messages
    )



##### STREAMLIT #####

st.title("Consciousness Science Bot")


### TEMPERATURE SLIDER ###
temp = st.slider('Adjust the bot\'s creativity level', 0.0, 2.0)


### PROMPT CUSTOMIZATION ###

# User can change the system prompts (see the dictionary above ^)
if new_prompt := st.selectbox('Choose an attitude for the bot:', ['Helpful', 'Unhelpful']):
    st.session_state['system_prompts'] = ['basic_context', new_prompt] #overwriting prompts instead of appending or swapping (for now)

### CHUNK NUMBER CUSTOMIZATION ###

st.sidebar.subheader("Context size")

chunk_levels = {
    "Small context": 2,
    "Medium context": 3,
    "Big context": 4,
}

selected_level = st.sidebar.radio(
    "How many chunks should be retrieved?",
    options=list(chunk_levels.keys()),
    index=1,  # default: Medium
)

num_chunks = chunk_levels[selected_level]
st.sidebar.caption(f"Retrieving {num_chunks} chunk(s) per query")

### CHAT ###

# Initializing chatbot
# If the parameters change, this reruns, otherwise it uses what is in the cache already
rag_bot = init_bot(
    prefix_messages=st.session_state['system_prompts'],
    temp=temp,
    num_chunks=num_chunks
)

# Display chat messages from history on app rerun
for message in rag_bot.chat_history:
    with st.chat_message(message.role):
        st.markdown(message.blocks[0].text)

# React to user input
if prompt := st.chat_input('Reset the chat by typing "Goodbye"'):

    # If user types "goodbye", reset the memory and run the app from the top again
    if prompt.lower() == 'goodbye':
        rag_bot.reset() # reset the bot memory
        st.rerun() # reruns the app so that the bot is reinitialized and the chat is cleared
    
    # Display user message in chat message container
    st.chat_message("human").markdown(prompt)

    # Begin spinner before answering question so it's there for the duration
    with st.spinner("Please wait for Marlo's Bot to compute the cosine similarities..."):
        # send question to bot to get answer
        answer = rag_bot.chat(prompt)

        # extract answer from bot's response
        response = answer.response

        # Display chatbot response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)