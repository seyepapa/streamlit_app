import streamlit as st
import ollama
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.llms.base import LLM  # Classe de base pour les modèles LangChain

# Définition d'une classe pour intégrer Ollama dans LangChain
class OllamaLLM(LLM):
    model: str = "mistral"  # Modèle par défaut

    def _call(self, prompt: str, stop=None) -> str:
        """Appelle le modèle Ollama pour générer une réponse."""
        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]

    @property
    def _llm_type(self) -> str:
        """Définit le type du modèle."""
        return "ollama"

# Configuration du prompt
prompt_template = PromptTemplate(
    input_variables=["history", "input"],
    template="""Vous êtes un assistant conversationnel intelligent. Voici l'historique de la conversation :
{history}

Utilisateur: {input}
Assistant:"""
)

# Initialisation de la mémoire et du modèle Ollama
memory = ConversationBufferMemory()
llm = OllamaLLM(model="mistral")  # Vous pouvez changer le modèle ("gemma", "llama2", etc.)

# Création de la chaîne de conversation
conversation = ConversationChain(
    prompt=prompt_template,
    memory=memory,
    llm=llm
)

# Interface Streamlit
st.title("AI assistant")
st.write("""Bienvenue sur notre site web, .""")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher les messages précédents
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrée utilisateur
if prompt := st.chat_input("Quelle est votre question ?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Générer la réponse de l'assistant avec Ollama
    response = conversation.predict(input=prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
