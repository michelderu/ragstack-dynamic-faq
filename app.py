import streamlit as st
import os
import tempfile
import asyncio

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import AstraDB
from langchain.schema.runnable import RunnableMap
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, WebBaseLoader
from langchain.cache import AstraDBSemanticCache
from langchain.globals import set_llm_cache

# Draw a title and some markdown
st.set_page_config(layout="wide")
st.image("./assets/postnl.webp", width=150)
st.title("Post NL Customer Care")
st.markdown("""Gebruik deze service om direct duidelijke antwoorden op je vragen te krijgen. Plus: een overzichtelijke lijst relevante veel gestelde vragen!""")
st.divider()

# Streaming call back handler for responses
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text + "▌")

# Function for Vectorizing uploaded data into Astra DB
def vectorize_text(uploaded_files, vector_store):
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            
            # Write to temporary file
            temp_dir = tempfile.TemporaryDirectory()
            file = uploaded_file
            print(f"""Processing: {file}""")
            temp_filepath = os.path.join(temp_dir.name, file.name)
            with open(temp_filepath, 'wb') as f:
                f.write(file.getvalue())

            # Process TXT
            if uploaded_file.name.endswith('txt'):
                file = [uploaded_file.read().decode()]

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size = 1500,
                    chunk_overlap  = 100
                )

                texts = text_splitter.create_documents(file, [{'source': uploaded_file.name}])
                vector_store.add_documents(texts)
                st.info(f"{len(texts)} chunks loaded")
            
            # Process PDF
            if uploaded_file.name.endswith('pdf'):
                docs = []
                loader = PyPDFLoader(temp_filepath)
                docs.extend(loader.load())

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size = 1500,
                    chunk_overlap  = 100
                )

                pages = text_splitter.split_documents(docs)
                vector_store.add_documents(pages)  
                st.info(f"{len(pages)} chunks loaded")

            # Process CSV
            if uploaded_file.name.endswith('csv'):
                docs = []
                loader = CSVLoader(temp_filepath)
                docs.extend(loader.load())

                vector_store.add_documents(docs)
                st.info(f"{len(docs)} chunks loaded")

# Load data from URLs
def vectorize_url(urls, vector_store):
    # Create the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1500,
        chunk_overlap  = 100
    )

    for url in urls:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()    
            pages = text_splitter.split_documents(docs)
            print (f"Loading from URL: {pages}")
            vector_store.add_documents(pages)  
            st.info(f"{len(pages)} loaded")
        except Exception as e:
            st.info(f"An error occurred:", e)


# Cache prompt for future runs
@st.cache_data()
def load_answer_prompt():
    template = """Je bent een vriendelijke AI assistent die gebruikers een duidelijk antwoord geeft dat concreet is.
Je antwoord uitgebreid en gebruikt opsommingstekens als het kan.
Als je het antwoord niet weet, zeg dat dan en verwijs naar www.postnl.nl.
Beantwoord vragen alleen met informatie over PostNL.

CONTEXT:
{context}

QUESTION:
{question}

YOUR ANSWER:"""
    return ChatPromptTemplate.from_messages([("system", template)])
answer_prompt = load_answer_prompt()

# Cache prompt for future runs
@st.cache_data()
def load_faq_prompt():
    template = """Maak een lijst met veel gestelde vragen en bijbehorende antwoorden. 
Beantwoord vragen alleen met informatie over PostNL.

Genereer alleen een lijst van vragen en antwoorden die direct gerelateerd zijn aan de volgende vraag:
{question}

Gebruik alleen de volgende context voor het maken van de lijst met veel gestelde vragen:
{context}

Het resultaat in de volgende structuur:
Begin met een heading: Relevante veel gestelde vragen
Gevolgd door repeterend de Vraag in bold en Antwoord in bold op de volgende regel"""
    return ChatPromptTemplate.from_messages([("system", template)])
faq_prompt = load_faq_prompt()

# Cache OpenAI Chat Model for future runs
@st.cache_resource()
def load_chat_model():
    return ChatOpenAI(
        temperature=0.3,
        model='gpt-4-turbo-preview',
        streaming=True,
        verbose=True
    )
chat_model = load_chat_model()

# Cache the Astra DB Vector Store for future runs
@st.cache_resource(show_spinner='Connecting to Astra')
def load_vector_store():
    # Connect to the Vector Store
    vector_store = AstraDB(
        embedding=OpenAIEmbeddings(),
        collection_name="postnl",
        api_endpoint=st.secrets['ASTRA_API_ENDPOINT'],
        token=st.secrets['ASTRA_TOKEN']
    )
    return vector_store
vector_store = load_vector_store()

# Cache the Retriever for future runs
@st.cache_resource(show_spinner='Getting retriever')
def load_retriever():
    # Get the retriever for the Chat Model
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )
    return retriever
retriever = load_retriever()

@st.cache_resource(show_spinner='Getting cache')
def set_cache():
    set_llm_cache(
        AstraDBSemanticCache(
            api_endpoint=st.secrets['ASTRA_API_ENDPOINT'],
            token=st.secrets['ASTRA_TOKEN'],
            embedding=OpenAIEmbeddings(),
            collection_name="semantic_cache",
        )
    )
#set_cache()

# Include the upload form for new data to be Vectorized
with st.sidebar:
    uploaded_files = st.file_uploader('Upload a document for additional context', type=['txt', 'pdf', 'csv'], accept_multiple_files=True)
    upload = st.button('Save to Astra DB', key='txt')
    if upload and uploaded_files:
        vectorize_text(uploaded_files, vector_store)

# Include the upload form for URLs be Vectorized
with st.sidebar:
    urls = st.text_area('Upload a URL for additional context', help='Separate multiple URLs with a comma (,)')
    urls = urls.split(',')
    upload = st.button('Save to Astra DB', key='url')
    if upload and urls:
        vectorize_url(urls, vector_store)

col1, col2 = st.columns(2)

with col1:
    question = st.text_input(
        "Wat is je vraag"
    )
    answer_placeholder = st.empty()

with col2:
    faq_placeholder = st.empty()

if question:
            
    # Create the runnable map with the context from the vector store
    inputs = RunnableMap({
        'context': lambda x: retriever.get_relevant_documents(x['question']),
        'question': lambda x: x['question']
    })

    # Generate the answer by calling OpenAI's Chat Model
    answer_chain = inputs | answer_prompt | chat_model
    # Generate the FAQ by calling OpenAI's Chat Model
    faq_chain = inputs | faq_prompt | chat_model

    # Draw the results to the screen
    answer_result = answer_chain.invoke({'question': question}, config={'callbacks': [StreamHandler(answer_placeholder)]})
    answer_placeholder.markdown(answer_result.content)

    faq_result = faq_chain.invoke({'question': question}, config={'callbacks': [StreamHandler(faq_placeholder)]})
    faq_placeholder.markdown(faq_result.content)