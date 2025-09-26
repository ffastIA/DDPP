import os  # Permite acessar arquivos, pastas e configurações do computador
import streamlit as st  # Framework para criar aplicações web interativas
from langchain_community.document_loaders import PyPDFLoader  # Carrega arquivos PDF
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Divide textos grandes em pedaços menores
from langchain_openai import OpenAIEmbeddings  # Converte texto em números (vetores)
from langchain_community.vectorstores import FAISS  # Banco de dados para armazenar e buscar vetores
from langchain_openai import ChatOpenAI  # Modelo de linguagem da OpenAI (GPT)
from langchain.chains import RetrievalQA  # Sistema de perguntas e respostas
from langchain.prompts import ChatPromptTemplate  # Template para formatar perguntas ao modelo
from dotenv import load_dotenv  # Carrega senhas do arquivo .env (arquivo secreto)
import tiktoken  # Conta quantas "palavras" (tokens) tem um texto
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.chat_history import InMemoryChatMessageHistory
import uuid  # Para gerar IDs únicos de sessão

# ============================ INICIA O STREAMLIT =====================================
st.set_page_config(
    page_title="IAgo",  # Título que aparece na aba do navegador
    page_icon="🧠",  # Ícone que aparece na aba
    layout="wide",  # Usa toda a largura da tela
    initial_sidebar_state = "collapsed"  # esconde a barra lateral por padrão
)
# =====================================================================================

# ============================ CHAVES DO SISTEMA ===============================
deploy = True # False == roda localmente / True == versão para deploy
cria_vector = False  # False == só carrega a vector store / True == cria a vector store
# ===============================================================================

# Tenta pegar a chave da API da OpenAI de duas formas diferentes
load_dotenv()
if not deploy:
    api_key = os.getenv("OPENAI_API_KEY")  # Do arquivo .env
else:
    api_key = st.secrets["OPENAI_API_KEY"]  # Do Streamlit secrets (para deploy)

# Verifica se conseguiu pegar a chave da API
if not api_key:
    raise ValueError("A variável de ambiente 'OPENAI_API_KEY' não foi encontrada no seu arquivo .env.")

# -----------------------------------------------------------------------------
# 1. DEFINIÇÕES GERAIS
# -----------------------------------------------------------------------------

# Define a chave API como variável de ambiente
os.environ["OPENAI_API_KEY"] = api_key

# Define o modelo de GPT
modelo = 'gpt-3.5-turbo-0125'  # Qual modelo do ChatGPT usar

# Cria uma instância do modelo de chat da OpenAI
chat_instance = ChatOpenAI(model=modelo)

# Define o modelo de embedding
embeddings_model = OpenAIEmbeddings()  # Modelo para converter texto em números

# Define o local onde o vector store persistirá
diretorio_vectorstore_faiss = 'vectorstore_faiss'  # Onde salvar o banco vetorial

# Define o diretório onde os arquivos para gerar o vector store estão localizados
caminho_arquivo = 'docs'  # Caminho dos arquivos para analisar

# Define a quantidade máxima de documentos retornados pela função retriever()
qtd_retriever = 4

# -----------------------------------------------------------------------------
# 1.5. CONFIGURAÇÃO DE MEMÓRIA CONVERSACIONAL
# -----------------------------------------------------------------------------

#Dicionário global para armazenar históricos por sessão
if 'memory_store' not in st.session_state:
    st.session_state.memory_store = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    Função que usa st.session_state para histórico
    """
    if session_id not in st.session_state.memory_store:
        st.session_state.memory_store[session_id] = InMemoryChatMessageHistory()
        print(f"Criada nova sessão {session_id}")  # Debug
    return st.session_state.memory_store[session_id]


def clear_session_history(session_id: str):
    """Limpa o histórico de uma sessão específica"""
    if session_id in st.session_state.memory_store:
        st.session_state.memory_store[session_id].clear()
        print(f"Limpeza da sessão {session_id}")  # Debug


def get_active_sessions():
    """Retorna lista de sessões ativas do session_state"""
    return list(st.session_state.memory_store.keys())


def get_session_message_count(session_id: str):
    """Retorna o número de mensagens usando session_state"""
    if session_id in st.session_state.memory_store:
        return len(st.session_state.memory_store[session_id].messages)
    return 0


#Função para testar e forçar salvamento
def force_save_test_message(session_id: str):
    """Força o salvamento de uma mensagem de teste"""
    history = get_session_history(session_id)
    history.add_user_message("TESTE: Mensagem de teste")
    history.add_ai_message("TESTE: Resposta de teste")
    return len(history.messages)


# -----------------------------------------------------------------------------
# 2. PROMPTS COM MEMÓRIA CONVERSACIONAL
# -----------------------------------------------------------------------------

# Template para pergunta com contexto conversacional
prompt_inicial = ChatPromptTemplate.from_messages([
    ("system", """Você é o IAgo, colaborador especializado no projeto DESPERTAR DIGITAL.
    Sua expertise é esclarecer dúvidas sobre este projeto de forma conversacional e personalizada.

    DIRETRIZES IMPORTANTES:
    - Use o contexto fornecido do documento para responder às perguntas
    - CONSIDERE o histórico da conversa para dar respostas mais personalizadas
    - Refira-se a perguntas anteriores quando relevante ("Como mencionei antes...", "Complementando sua pergunta anterior...")
    - Você pode buscar dados complementares na internet, mas SEMPRE forneça as fontes
    - Se não souber a resposta, seja transparente e diga que não sabe
    - NUNCA invente informações
    - Mantenha um tom conversacional e amigável
    - Adapte suas respostas baseado no nível de conhecimento demonstrado pelo usuário
    - Se for a primeira pergunta da conversa, apresente-se brevemente

    CONTEXTO DO DOCUMENTO:
    {context}"""),

    ("placeholder", "{chat_history}"),  # Aqui será inserido o histórico

    ("human", "{question}"),
])

# Template para tradução com contexto conversacional
translation_prompt = ChatPromptTemplate.from_messages([
    ("system", """Você é um tradutor especializado em textos técnicos sobre aquicultura e biometria.

    INSTRUÇÕES:
    - Traduza mantendo precisão técnica
    - Preserve formatação e termos técnicos específicos
    - Considere o contexto conversacional se houver referências anteriores
    - Mantenha naturalidade em inglês
    - Se houver referências a conversas anteriores, traduza-as adequadamente"""),

    ("placeholder", "{chat_history}"),  # Histórico para contexto de tradução

    ("human", "Traduza este texto: {text}"),
])


# -----------------------------------------------------------------------------
# 3. FUNÇÕES
# -----------------------------------------------------------------------------

def cria_vector_store_faiss(chunk: list[Document], diretorio_vectorestore_faiss: str) -> FAISS:
    vector_store = FAISS.from_documents(chunk, embeddings_model)
    vector_store.save_local(diretorio_vectorestore_faiss)
    return vector_store


def carrega_vector_store_faiss(diretorio_vectorestore_faiss, embedding_model):
    vector_store = FAISS.load_local(
        diretorio_vectorestore_faiss,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return vector_store


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def cria_chunks(caminho: str, chunk_size: int, chunk_overlap: int) -> list:
    print(f"Carregando documentos do PDF: {caminho_arquivo}")

    loader = DirectoryLoader(
        path=caminho,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        recursive=False
    )

    documentos = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=num_tokens_from_string,
        separators=['&', '\n\n', '.', ' '],
        add_start_index=True
    )

    chunk = text_splitter.split_documents(documentos)
    print(f"Texto original dividido em {len(chunk)} chunks.\n")
    return chunk


def format_docs(docs: list[Document]):
    return "\n\n".join(doc.page_content for doc in docs)


def retriever(pergunta: str, n: int):
    resultado = vectorstore.as_retriever(search_type='mmr', search_kwargs={"k": n})
    documentos_retornados = resultado.get_relevant_documents(pergunta)
    return documentos_retornados


# -----------------------------------------------------------------------------
# 4. VECTOR STORE
# -----------------------------------------------------------------------------

if cria_vector:
    chunks = cria_chunks(caminho_arquivo, 500, 50)
    vectorstore = cria_vector_store_faiss(chunks, diretorio_vectorstore_faiss)
else:
    vectorstore = carrega_vector_store_faiss(diretorio_vectorstore_faiss, embeddings_model)

# -----------------------------------------------------------------------------
# 6. TÍTULO E DESCRIÇÃO DA PÁGINA PELO STREAMLIT
# -----------------------------------------------------------------------------

st.title("🧠 PROJETO DESPERTAR DIGITAR")
st.markdown("""
Esta aplicação permite que você consulte os projetos Despertar Digital usando Inteligência Artificial **com memória conversacional**.
Faça perguntas sobre seus objetivos, indicadores e benefícios! 
A IA se lembrará das suas perguntas anteriores para dar respostas mais personalizadas.
""")


# ================================== PIPELINE COM MEMÓRIA  =======================================
# VERSÃO FINAL - COM HISTÓRICO FORÇADO
# ================================================================================================

#Função auxiliar para buscar contexto
def get_context_for_question(question: str) -> str:
    """Busca documentos relevantes e formata como contexto"""
    docs = retriever(question, qtd_retriever)
    return format_docs(docs)


#Função que inclui histórico manualmente no prompt
def create_rag_response_with_history(question: str, session_id: str) -> str:
    """
    Função que cria resposta RAG COM HISTÓRICO MANUAL
    """
    # Busca o contexto dos documentos
    context = get_context_for_question(question)

    #Pega o histórico manualmente
    history = get_session_history(session_id)

    # Formata o histórico para incluir no prompt
    chat_history_text = ""
    if history.messages:
        chat_history_text = "\n\nHISTÓRICO DA CONVERSA:\n"
        for msg in history.messages[-10:]:  # Últimas 10 mensagens
            if msg.type == "human":
                chat_history_text += f"👤 Usuário: {msg.content}\n"
            else:
                chat_history_text += f"🤖 Assistente: {msg.content}\n"
        chat_history_text += "\n"

    # Monta o prompt completo com histórico
    full_prompt = f"""Você é o IAgo, colaborador especializado no projeto DESPERTAR DIGITAL da SETEC e do Instituto Idear.
        Sua expertise é esclarecer dúvidas sobre este projeto de forma conversacional e personalizada.
        
        DIRETRIZES IMPORTANTES:
        - Use o contexto fornecido do documento para responder às perguntas
        - CONSIDERE o histórico da conversa para dar respostas mais personalizadas
        - Refira-se a perguntas anteriores quando relevante ("Como mencionei antes...", "Complementando sua pergunta anterior...")
        - Você pode buscar dados complementares na internet, mas SEMPRE forneça as fontes
        - Se não souber a resposta, seja transparente e diga que não sabe
        - NUNCA invente informações
        - Mantenha um tom conversacional e amigável
        - Adapte suas respostas baseado no nível de conhecimento demonstrado pelo usuário
        - Se for a primeira pergunta da conversa, apresente-se brevemente
        
        CONTEXTO DO DOCUMENTO:
        {context}
        
        {chat_history_text}
        
        PERGUNTA ATUAL: {question}
        
        Responda considerando todo o contexto e histórico acima:"""

    # Executa o prompt
    response = chat_instance.invoke(full_prompt)
    return response.content


#Função de tradução com histórico manual
def create_translation_with_history(text: str, session_id: str) -> str:
    """
    Função que traduz COM HISTÓRICO MANUAL
    """
    #Pega o histórico manualmente
    history = get_session_history(session_id)

    # Formata o histórico para contexto de tradução
    chat_history_text = ""
    if history.messages:
        chat_history_text = "\n\nCONTEXTO DA CONVERSA (para referência):\n"
        for msg in history.messages[-4:]:  # Últimas 4 mensagens
            if msg.type == "human":
                chat_history_text += f"Usuário perguntou: {msg.content[:100]}...\n"
            else:
                chat_history_text += f"Assistente respondeu: {msg.content[:100]}...\n"

    # Monta o prompt de tradução
    translation_prompt_text = f"""Você é um tradutor especializado em textos técnicos sobre projetos sociais.
        
        INSTRUÇÕES:
        - Traduza para a lingua inglesa mantendo precisão técnica
        - Preserve formatação e termos técnicos específicos
        - Considere o contexto conversacional se houver referências anteriores
        - Mantenha naturalidade em inglês
        - Se houver referências a conversas anteriores, traduza-as adequadamente
        
        {chat_history_text}
        
        TEXTO PARA TRADUZIR:
        {text}
        
        Traduza para inglês:"""

    # Executa a tradução
    response = chat_instance.invoke(translation_prompt_text)
    return response.content


#Função principal COMPLETAMENTE REESCRITA
def qa_chain_complete_with_memory(query: str, session_id: str):
    """Função FINAL com histórico manual garantido"""

    try:
        print(f"Processando pergunta: {query}")
        print(f"Session ID: {session_id}")

        #Pega o histórico ANTES de processar
        history = get_session_history(session_id)
        print(f"Mensagens no histórico ANTES: {len(history.messages)}")

        #Mostra o conteúdo do histórico para debug
        if history.messages:
            print("Últimas mensagens do histórico:")
            for i, msg in enumerate(history.messages[-2:]):
                print(f"  {i + 1}. {msg.type}: {msg.content[:50]}...")

        # Primeiro: gera resposta COM HISTÓRICO MANUAL
        resposta_original = create_rag_response_with_history(query, session_id)

        # Segundo: traduz COM HISTÓRICO MANUAL
        resposta_traduzida = create_translation_with_history(resposta_original, session_id)

        # 🔧 CORREÇÃO: SALVA AS MENSAGENS NO HISTÓRICO
        history.add_user_message(query)
        history.add_ai_message(resposta_original)

        print(f"Mensagens no histórico DEPOIS: {len(history.messages)}")

        # Terceiro: busca documentos fonte
        source_docs = retriever(query, qtd_retriever)

        print(f"Resposta gerada com sucesso")

        return {
            "result": resposta_original,
            "translated": resposta_traduzida,
            "source_documents": source_docs,
            "session_id": session_id
        }

    except Exception as e:
        print(f"Erro detalhado: {type(e).__name__}: {str(e)}")
        st.error(f"Erro ao processar pergunta: {str(e)}")

        #Fallback simples MAS com salvamento
        try:
            st.warning("Tentando processar sem contexto avançado...")

            # Resposta simples
            context = get_context_for_question(query)
            simple_response = chat_instance.invoke(
                f"Baseado no contexto sobre os projetos Despetar Digital , responda de forma amigável: {query}\n\nContexto: {context}"
            ).content

            # Tradução simples
            simple_translation = chat_instance.invoke(
                f"Traduza para inglês mantendo o tom técnico: {simple_response}"
            ).content

            # 🔧 CORREÇÃO: Salva no histórico mesmo no fallback
            history = get_session_history(session_id)
            history.add_user_message(query)
            history.add_ai_message(simple_response)

            print(f"Fallback executado, histórico tem {len(history.messages)} mensagens")

            return {
                "result": simple_response,
                "translated": simple_translation,
                "source_documents": retriever(query, qtd_retriever),
                "session_id": session_id
            }

        except Exception as fallback_error:
            print(f"Erro no fallback: {str(fallback_error)}")
            return {
                "result": "Desculpe, ocorreu um erro técnico. Tente reformular sua pergunta.",
                "translated": "Sorry, a technical error occurred. Please try rephrasing your question.",
                "source_documents": [],
                "session_id": session_id
            }


# ================================================
# 10. INTERFACE DE CHAT COM MEMÓRIA
# ================================================

st.markdown("---")
#st.header("💬 Converse com o projeto BIA!")
st.header("💬 Converse com o consultor virtual ->  IAgo!")

#Inicializa session_id e garante que seja adicionado ao store
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    #Força a criação da sessão no store
    get_session_history(st.session_state.session_id)

# Sidebar para gerenciar sessões e memória
with st.sidebar:
    st.subheader("🧠 Gerenciamento de Memória")

    # Mostra ID da sessão atual (primeiros 8 caracteres)
    st.text(f"Sessão: {st.session_state.session_id[:8]}...")

    #Mostra estatísticas da sessão com refresh automático
    message_count = get_session_message_count(st.session_state.session_id)
    st.metric("Mensagens na memória", message_count)

    #Botão para forçar atualização
    if st.button("🔄 Atualizar Contadores"):
        st.rerun()

    # Botão para limpar memória da sessão atual
    if st.button("🗑️ Limpar Memória", help="Limpa o histórico conversacional da IA"):
        clear_session_history(st.session_state.session_id)
        st.session_state.messages = []
        st.success("Memória conversacional limpa!")
        st.rerun()

    # Botão para nova sessão
    if st.button("🆕 Nova Sessão", help="Inicia uma nova conversa independente"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        # 🔧 CORREÇÃO: Força a criação da nova sessão no store
        get_session_history(st.session_state.session_id)
        st.success("Nova sessão iniciada!")
        st.rerun()

    #Mostra número de sessões ativas
    active_sessions = get_active_sessions()
    st.info(f"Sessões ativas: {len(active_sessions)}")

    #Debug das sessões MELHORADO
    with st.expander("🔧 DEBUG - Detalhes da Memória"):
        st.write("**Sessões no store:**", active_sessions)
        st.write("**Mensagens no Streamlit:**", len(st.session_state.messages) if "messages" in st.session_state else 0)

        if st.session_state.session_id in st.session_state.memory_store:
            history = st.session_state.memory_store[st.session_state.session_id]
            st.write(f"**Mensagens no LangChain:** {len(history.messages)}")

            #Mostra as últimas mensagens da memória
            if history.messages:
                st.write("**Últimas mensagens na memória:**")
                for i, msg in enumerate(history.messages[-4:]):  # Últimas 4 mensagens
                    role = "👤 User" if msg.type == "human" else "🤖 AI"
                    content = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
                    st.text(f"{role}: {content}")
        else:
            st.write("**Sessão não encontrada no store!**")

    #Botão para testar memória MELHORADO
    if st.button("🧪 Testar Memória"):
        history = get_session_history(st.session_state.session_id)
        st.write(f"**Teste Atual:** {len(history.messages)} mensagens encontradas")

        if history.messages:
            st.write("**Conteúdo das mensagens:**")
            for i, msg in enumerate(history.messages):
                role = "👤 User" if msg.type == "human" else "🤖 AI"
                content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                st.text(f"{i + 1}. {role}: {content}")
        else:
            st.write("**Nenhuma mensagem encontrada!**")

        # Botão para forçar teste
        if st.button("🔧 Forçar Teste", key="force_test"):
            count = force_save_test_message(st.session_state.session_id)
            st.success(f"Mensagem de teste adicionada! Total: {count}")
            st.rerun()

    # Informações sobre memória
    with st.expander("ℹ️ Sobre a Memória"):
        st.markdown("""
        **Como funciona:**
        - A IA se lembra das perguntas anteriores
        - Respostas são contextualizadas com a conversa
        - Cada sessão mantém seu próprio histórico
        - Use "Limpar Memória" para recomeçar
        - Use "Nova Sessão" para conversa independente

        **🔧 Debug:**
        - Use "Atualizar Contadores" se os números não baterem
        - Use "Testar Memória" para ver o conteúdo
        - Use "Forçar Teste" para adicionar mensagens de teste
        """)

# Inicializa o histórico de mensagens se não existir
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostra o histórico de mensagens
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            tab1, tab2 = st.tabs(["🇧🇷 Português", "🇺🇸 SOBRAL"])

            with tab1:
                st.markdown(msg["content"])

            with tab2:
                if "translated" in msg:
                    st.markdown(msg["translated"])
                else:
                    st.info("Tradução não disponível para mensagens anteriores")

            if "sources" in msg and msg["sources"]:
                with st.expander("📚 Fontes"):
                    for i, source in enumerate(msg["sources"], 1):
                        st.markdown(f"**Fonte {i}:**")
                        st.markdown(source.page_content[:200] + "...")
                        st.markdown(f"*Página: {source.metadata.get('page', 'N/A')}*")
                        st.markdown("---")

# Campo de entrada para nova pergunta
user_input = st.chat_input(
    "Digite sua pergunta sobre o documento... (A IA se lembra das perguntas anteriores)",
    key="user_input_field"
)

# 🔧 CORREÇÃO: Processa nova pergunta COM MEMÓRIA MANUAL
if user_input:
    # Adiciona pergunta do usuário ao histórico do Streamlit
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Mostra a pergunta do usuário
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Processa com memória conversacional
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🧠 Analisando com contexto da conversa..."):
            # 🔧 CORREÇÃO: USA A FUNÇÃO COM HISTÓRICO MANUAL FORÇADO
            result = qa_chain_complete_with_memory(
                user_input,
                st.session_state.session_id
            )

            answer = result["result"]
            translated = result["translated"]
            sources = result.get("source_documents", [])

        # Mostra as respostas em abas
        tab1, tab2 = st.tabs(["🇧🇷 Português", "🇺🇸 SOBRAL"])

        with tab1:
            st.markdown(answer)

        with tab2:
            st.markdown(translated)

        # Mostra as fontes
        if sources:
            with st.expander("📚 Fontes"):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"**Fonte {i}:**")
                    st.markdown(source.page_content[:200] + "...")
                    st.markdown(f"*Página: {source.metadata.get('page', 'N/A')}*")
                    st.markdown("---")

        # Adiciona resposta ao histórico do Streamlit
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "translated": translated,
            "sources": sources
        })
