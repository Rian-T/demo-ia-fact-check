import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import time
from dotenv import load_dotenv
import os
load_dotenv()

# Configuration de la page Streamlit
st.set_page_config(page_title="IA de Fact-Checking pour Simon Puech", page_icon="🔍", layout="wide")

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

st.title("🔍 IA de Fact-Checking pour Simon Puech")
st.write("J'étais sur le stream de Simon Puech, il parlait d'une IA pour fact-checker des informations pendant un stream.")
st.write("J'ai trouvé l'idée cool donc j'ai fais un proto")
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .expander-content {
        font-size: 16px;
        line-height: 1.6;
    }
    .expander-title {
        font-size: 18px;
        font-weight: bold;
        color: #4CAF50;
    }
    .expander-subtitle {
        font-size: 16px;
        font-weight: bold;
        color: #333;
    }
    .expander-text {
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialisation de ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

# Fonction pour formater les documents
def format_docs(docs):
    retrieved = ""
    for doc in docs:
        title = doc.metadata["title"]
        source = doc.metadata["source"]
        retrieved += f"Titre:\n{title}\n\nContenu:\n{doc.page_content}\n\nSource:\n{source}\n\n"
    return retrieved

# Prompt pour générer des mots-clés et choisir la langue
keyword_prompt = ChatPromptTemplate.from_template(
    """
    Based on the following question, generate a list of 1-5 relevant keywords or phrases for searching on Wikipedia.
    Also, choose the most appropriate language for this search. Consider factors like:
    - The subject matter (e.g., French might be best for topics specific to France)
    - The potential for more comprehensive information (e.g., English often has more extensive articles)
    - Another language than english should be chosen only if country-specific information is needed.
    - Except for English, don't choose a language that is not precisely related to the question.
    - Question will always be in French, so don't be biased by the language of the question.
    - They should be different enough from previous keywords to expand the search.

    Question: {question}
    Previously used languages: {previous_languages}
    Previously used keywords: {previous_keywords}

    Provide your response in the following format:
    Language: [two-letter language code, e.g., 'en', 'fr', 'es', 'de', 'it']
    Keywords:
    - [keyword 1]
    - [keyword 2]
    - [keyword 3]
    Explanation: [Brief explanation in French for your language choice and keywords]
    """
)

# Prompt pour analyser les informations récupérées et décider des prochaines étapes
analysis_prompt = ChatPromptTemplate.from_template(
    """
    Analyze the following information retrieved from Wikipedia based on the question.
    Determine if the information is sufficient to answer the question or if more information is needed.
    You should only speak in French.

    Question: {question}
    Retrieved Information:
    {retrieved_info}

    1. Summarize the key points from the retrieved information, provide every relevant numbers precisely.
    2. Identify any gaps in the information or areas that need clarification.
    3. Decide if more information is needed. If yes, suggest what kind of information to look for next.
    4. If the information is sufficient, provide a concise answer to the question in French.

    Format your response in raw text as follows:
    Résumé: [Your summary here]
    Manque d'infos: [Identified gaps or areas needing clarification]
    Qu'en penser: [More information needed / Sufficient information]
    Prochaine recherche: [Suggestions for next search]
    Réponse à ce stade: [Your answer in French here]
    """
)

# Prompt pour la réponse finale
final_answer_prompt = ChatPromptTemplate.from_template(
    """
    Based on all the information gathered, provide a comprehensive answer to the question in French. Follow these guidelines:

    1. Summarize the key points from all retrieved information.
    2. Provide a well-reasoned, cautious answer that avoids overconfidence.
    3. If there are conflicting pieces of information, highlight them and explain how they affect your confidence level.
    4. Cite sources where appropriate.
    5. Use the following format for your response:

    ```markdown
    ### Résumé des informations clés :
    - Point 1
    - Point 2
    - ...

    ### Analyse et raisonnement :
    [Votre analyse détaillée ici]

    ### Réponse finale :
    [Votre réponse concise en français]

    ### Niveau de confiance :
    [Indiquez votre niveau de confiance : Élevé / Modéré / Faible]

    ### Sources :
    - [Source 1]
    - [Source 2]
    - ...
    ```

    Question: {question}
    
    Retrieved Information:
    {all_retrieved_info}
    """
)

def iterative_search(question):
    all_retrieved_info = ""
    all_analysis = ""
    iteration = 0
    max_iterations = 3
    previous_keywords = []
    previous_languages = []
    max_context_length = 200000  # Adjust this value based on your model's context limit

    progress_bar = st.progress(0)
    status_text = st.empty()

    while iteration < max_iterations:
        progress_bar.progress((iteration + 1) / (max_iterations + 1))
        
        status_text.text(f"Itération {iteration + 1} : Génération des mots-clés et choix de la langue")
        time.sleep(1)  # Pour l'effet visuel
        
        # Générer des mots-clés et choisir la langue
        keywords_response = llm.invoke(keyword_prompt.format(
            question=question,
            previous_languages=", ".join(previous_languages),
            previous_keywords=", ".join(previous_keywords)
        ))
        response_lines = keywords_response.content.split("\n")
        language = response_lines[0].split(": ")[1].strip()
        keywords = [line.strip("- ") for line in response_lines[1:-1] if line.startswith("-")]
        explanation = response_lines[-1].split(": ")[1].strip()
        
        previous_languages.append(language)
        previous_keywords.extend(keywords)
        
        with st.expander(f"📚 Itération {iteration + 1} - Recherche d'information - Mots-clés et Langue", expanded=True):
            st.markdown(f"""
                <div class="expander-content">
                    <div class="expander-title">Langue choisie :</div>
                    <div class="expander-text">{language}</div>
                    <div class="expander-title">Mots-clés :</div>
                    <div class="expander-text">{", ".join(keywords)}</div>
                    <div class="expander-title">Explication :</div>
                    <div class="expander-text">{explanation}</div>
                </div>
            """, unsafe_allow_html=True)

        status_text.text(f"Itération {iteration + 1} : Récupération des informations")
        time.sleep(1)  # Pour l'effet visuel
        
        # Récupérer les informations
        all_docs = []
        retriever = WikipediaRetriever(top_k_results=3, lang=language, doc_content_chars_max=8000)
        for keyword in keywords:
            docs = retriever.get_relevant_documents(keyword)
            all_docs.extend(docs)
        
        retrieved_info = format_docs(all_docs)

        # Manage context length
        if len(all_retrieved_info) + len(retrieved_info) > max_context_length:
            # If adding new info exceeds the limit, keep only the most recent relevant information
            all_retrieved_info = all_retrieved_info[-(max_context_length - len(retrieved_info)):] + retrieved_info
        else:
            all_retrieved_info += retrieved_info

        with st.expander(f"💾 Itération {iteration + 1} - Informations récupérées brutes", expanded=False):
            st.markdown(all_retrieved_info)

        status_text.text(f"Itération {iteration + 1} : Analyse des informations")
        time.sleep(1)  # Pour l'effet visuel
        
        # Analyser les informations récupérées
        analysis_response = llm.invoke(analysis_prompt.format(
            question=question,
            retrieved_info=retrieved_info
        ))
        analysis = analysis_response.content
        all_analysis += analysis

        with st.expander(f"🧠 Itération {iteration + 1} - Analyse des résultats", expanded=True):
            st.markdown(f"""
                <div class="expander-content">
                    <div class="expander-subtitle">Analyse :</div>
                    <div class="expander-text">{analysis}</div>
                </div>
            """, unsafe_allow_html=True)

        if "Sufficient information" in analysis:
            break

        iteration += 1

    status_text.text("Génération de la réponse finale")
    progress_bar.progress(1.0)
    
    # Générer la réponse finale
    final_answer_response = llm.invoke(final_answer_prompt.format(
        question=question,
        all_retrieved_info=all_retrieved_info
    ))
    return final_answer_response.content

st.markdown('<p class="big-font">Posez votre question :</p>', unsafe_allow_html=True)
question = st.text_area("", "Est-ce que le tabac fait plus de dégats à la société que l'alcool ?")

if st.button("Soumettre", type="primary"):
    with st.spinner("Recherche en cours..."):
        final_answer = iterative_search(question)
    st.success("Recherche terminée !")
    try:
        final_answer = final_answer.split("```markdown")[1].split("```")[0].strip()
    except:
        pass
    st.markdown(final_answer)

# Pied de page
st.markdown("---")
st.markdown("Prototype développé par @superi4n en ~3 heures (améliorable de zinzin)")