##############################################
# ptdr le code est horrible mais ça marche   #
##############################################

import json
import os
import re
from typing import List
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.output_parsers import PydanticOutputParser
from langchain.utilities import GoogleSearchAPIWrapper
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from newspaper import Article
from pydantic import BaseModel, Field, model_validator

load_dotenv(override=True)


class TextFormatter(BaseModel):
    text_formatted: str = Field(description="The formatted text with highlights")

text_formatter_parser = PydanticOutputParser(pydantic_object=TextFormatter)

class RephraseResponse(BaseModel):
    evaluation: str = Field(
        description="Your evaluation of the original question in French"
    )
    necessary_rephrasing: bool = Field("Whether rephrasing is necessary")
    rephrased_question: str = Field(
        description="Your rephrased question or original question in French"
    )

rephrase_parser = PydanticOutputParser(pydantic_object=RephraseResponse)

class AnalysisItem(BaseModel):
    title: str = Field(description="Title of the analysis item")
    content: str = Field(description="Comprehensive and detailed analysis")
    source: str = Field(description="Source of the analysis item")

class ChooseSourceResponse(BaseModel):
    analysis: List[AnalysisItem] = Field(description="Analysis for each search result")
    chosen_results: List[int] = Field(description="Indices of chosen results")
    reason: str = Field(description="Reason for choosing this source in French")

choose_source_parser = PydanticOutputParser(pydantic_object=ChooseSourceResponse)

class FoundInformationItem(BaseModel):
    summary: str = Field(description="Summary of the found information item")
    quotes: List[str] = Field(description="exact quotes from the content")
    source: str = Field(
        description="Source of the found information item, ideally the direct URL"
    )

class AnalyzeContentResponse(BaseModel):
    found_information: List[FoundInformationItem] = Field(
        description="Found information in the content"
    )
    current_analysis: str = Field(description="Current analysis in French")
    need_more_info: bool = Field(description="Whether more information is needed")
    additional_info_needed: str = Field(
        description="Additional information needed in French"
    )
    new_google_query: str = Field(description="New Google query to search for")

analyze_content_parser = PydanticOutputParser(pydantic_object=AnalyzeContentResponse)


def top5_results(query):
    return search.results(query, 10)

def extract_domain(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc

def fetch_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article
    except Exception as e:
        return f"Erreur lors de l'extraction: {str(e)}"

st.set_page_config(
    page_title="IA de Fact-Checking Améliorée", page_icon="🔍", layout="wide"
)

st.title("🔍 IA de Fact-Checking pour Simon Puech V2")
st.write("J'étais sur le discord de Simon Puech, @Sekari avait de bonnes idées")
st.write("Alors j'ai implementé une V2")

st.markdown(
    """
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .highlight {
        background-color: #358205;  
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 5px;
    }
    .card {
        border: 1px solid #ddd;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        border-radius: 8px;
        margin-bottom: 20px;
        background-color: #fff;
    }
    .card h3 {
        font-size: 1.8em;
        color: #333;
    }
    .card-content {
        font-size: 1.1em;
        line-height: 1.6;
    }
    .card p {
        margin: 10px 0;
        color: #333;
    }
    .card:hover {
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    .card a {
        color: #1a73e8;
        text-decoration: none;
    }
    .card a:hover {
        text-decoration: underline;
    }
    .card-image {
        width: 100%;
        height: 150px;
        object-fit: contain;
        float: left; 
        margin: 0px 15px 15px 0px;
    }
      .info-summary {
        font-size: 1.1em;
        font-weight: 500;
        color: #333;
    }
    .info-source {
        font-size: 0.9em;
        font-weight: bold;
        color: #1a73e8;
    }
    .info-quote {
        font-size: 0.9em;
        font-style: italic;
        color: #555;
    }

    mark {
        background-color: yellow;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize handlers and API
search = GoogleSearchAPIWrapper(
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    google_cse_id=os.getenv("GOOGLE_CSE_ID"),
)

rephrase_prompt = ChatPromptTemplate.from_template(
    """
    I need to perform a Google search to answer this question: {question}
    
    To get the best results, I may need to rephrase the question.
    If stricly necessary, suggest a rephrased version more suitable for web search.
    You can return the original question if it's already okay. Don't do minor edits.
    Keep it mind that it is for google search, so don't change it for a more elaborate question.
    Remember that any text shown to the user should be in French.
    
    {format_instructions}
"""
)

choose_source_prompt = ChatPromptTemplate.from_template(
    """
    Question to answer: {question}
    
    Available search results:
    {search_results}
    
    Analyze each result and choose the most relevant and reliable source to answer our question.
    Explain your reasoning.
    Consider the quality of the content, the authority of the source, and the relevance to the question.
    If it is a sensitive topic, try to choose multiple sources from different points of view.
    Remember that any text shown to the user should be in French.
    
    {format_instructions}
"""
)

text_formatter_prompt = ChatPromptTemplate.from_template(
    """
    Question to answer: {question}
    
    Text to format: 
    -------------------
    {text}
    -------------------

    Please keep only the 3-4 paragraphs that are most relevant to the question separated by a new line.
    Please highlight using <mark> </mark> the very very most important parts of the text that answer the question.
    
    {format_instructions}
"""
)

analyze_content_prompt = ChatPromptTemplate.from_template(
    """
    Question: {question}
    
    Previous query sent to Google:
    {previous_question}

    Previous analysis:
    {previous_analysis}

    Content to analyze:
    {content}
    
    Analyze this content and determine:
    1. What relevant information it contains to answer the question
    2. If we need more information
    
    Remember that any text shown to the user should be in French.
    
    {format_instructions}
"""
)

previous_question = ""
all_analysis = []
all_urls = []

def process_question(question, max_iterations=3):
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    progress_bar = st.progress(0)
    iteration = 0
    all_content = []
    previous_question = question
    # Rephrase question
    with st.expander("🤔 Reformulation de la question", expanded=True):
        rephrase_response = llm.invoke(
            rephrase_prompt.format(
                question=question,
                format_instructions=rephrase_parser.get_format_instructions(),
            )
        )
        try:
            rephrase_data = rephrase_parser.invoke(rephrase_response.content)
            st.markdown(f"**Évaluation:** {rephrase_data.evaluation}")
            if rephrase_data.necessary_rephrasing:
                st.markdown(
                    f"**Question reformulée:** {rephrase_data.rephrased_question}"
                )
                search_question = rephrase_data.rephrased_question
            else:
                search_question = question
        except json.JSONDecodeError:
            st.error("Erreur lors du traitement de la reformulation")
            return None

    while iteration < max_iterations:
        progress_bar.progress((iteration + 1) / max_iterations)

        results = top5_results(search_question)
        # remove in results if already in all_urls
        results = [r for r in results if r["link"] not in all_urls]

        # Choose source
        with st.expander(
            f"🎣 Récupération d'articles - Itération {iteration + 1}",
            expanded=False,
        ):
            results_formatted = "\n".join(
                [
                    f"{i+1}. {r['title']}\n   Snippet: {r['snippet']}\n   URL: {r['link']}"
                    for i, r in enumerate(results)
                ]
            )

            for i, result in enumerate(results):
                st.markdown(
                    f"""
                    <b>{i+1}. {result['title']}</b><br>
                    Snippet: {result['snippet']}<br>
                    <a href="{result['link']}">{result['link']}</a>
                """,
                    unsafe_allow_html=True,
                )
            choice_response = llm.invoke(
                choose_source_prompt.format(
                    question=question,
                    search_results=results_formatted,
                    format_instructions=choose_source_parser.get_format_instructions(),
                )
            )

            try:

                choice_data = choose_source_parser.invoke(choice_response.content)

                st.markdown("#### Analyse des résultats:")
                for analysis in choice_data.analysis:
                    print(analysis)
                    st.markdown(f"**{analysis.title}:** {analysis.content}")

                st.markdown(
                    f"**Nous allons analyser les articles:** {choice_data.chosen_results}"
                )
                st.markdown(f"**Raison:** {choice_data.reason}")

                chosen_indices = [index - 1 for index in choice_data.chosen_results]
            except json.JSONDecodeError:
                st.error("Erreur lors du traitement de l'analyse des résultats")
                return None

        with st.expander(
            f"🔍 Analyse des résultats - Itération {iteration + 1}",
            expanded=True,
        ):
            for i, result in enumerate(results):
                if i in chosen_indices:
                    all_urls.append(result["link"])
                    article = fetch_article_content(result["link"])
                    try:
                        title = article.title
                        text = article.text
                        text_formatted = llm.invoke(
                            text_formatter_prompt.format(
                                text=text,
                                question=question,
                                format_instructions=text_formatter_parser.get_format_instructions(),
                            )
                        )
                        text_formatted = text_formatter_parser.invoke(
                            text_formatted.content
                        ).text_formatted
                        all_analysis.append(f"**{title}**\n{text_formatted}\n\n")
                        top_image = article.top_image
                        authors = ", ".join(article.authors)

                        html_content = f"""
                        <div class="card">
                            <div class="card-content">
                                <h3>{result["title"]}</h3>
                                <a href="{result['link']}">{result['link']}</a>
                                <p>{text_formatted}</p>
                            </div>
                        </div>
                        """

                        # Add authors if not empty
                        # if authors:
                        #     html_content += f"<p><b>Auteurs:</b> {authors}</p>"

                        # Display the content
                        st.markdown(html_content, unsafe_allow_html=True)
                    except Exception as e:
                        print(e)
                else:
                    pass
                    # st.markdown(
                    #     f"""
                    #     <b>{result['title']}</b><br>
                    #     {result['snippet']}<br>
                    #     <a href="{result['link']}">{result['link']}</a>
                    # """,
                    #     unsafe_allow_html=True,
                    # )

        all_content = []
        for chosen_index in chosen_indices:
            chosen_result = results[chosen_index]
            content = fetch_article_content(chosen_result["link"])
            all_content.append(
                {
                    "title": chosen_result["title"],
                    "link": chosen_result["link"],
                    "content": content,
                }
            )

        with st.expander(
            f"📚 Analyse du contenu - Itération {iteration + 1}", expanded=True
        ):
            full_content = "\n\n".join(
                [f"[{c['title']}]({c['link']}):\n{c['content']}" for c in all_content]
            )
            analysis_response = llm.invoke(
                analyze_content_prompt.format(
                    question=question,
                    previous_question=previous_question,
                    previous_analysis=all_analysis,
                    content=full_content,
                    format_instructions=analyze_content_parser.get_format_instructions(),
                )
            )

            try:

                analysis_data = analyze_content_parser.invoke(analysis_response.content)
                previous_question += " " + question
                question = analysis_data.new_google_query
                all_urls.append(
                    f"**Analyse iteration {iteration + 1}**\n{analysis_data.current_analysis}"
                )

                st.markdown("### Informations trouvées:")
                for info in analysis_data.found_information:
                    st.markdown(
                        f'<div class="info-summary">{info.summary}</div>',
                        unsafe_allow_html=True,
                    )
                    all_analysis.append(f"{info.summary}")

                    domain = extract_domain(info.source)

                    st.markdown(
                        f'<div class="info-source">Source: <a href="{info.source}" target="_blank">{domain}</a></div>',
                        unsafe_allow_html=True,
                    )

                    all_analysis.append(f"**Source:** {info.source}")
                    for quote in info.quotes:
                        st.markdown(
                            f'<div class="info-quote">- "{quote}"</div>',
                            unsafe_allow_html=True,
                        )
                        all_analysis.append(f'- "{quote}"')

                st.markdown(f"### Analyse actuelle:")
                st.markdown(analysis_data.current_analysis)
                all_analysis.append(
                    f"**Analyse iteration {iteration + 1}**\n{analysis_data.current_analysis}"
                )

                full_analysis = f"Round {iteration + 1}:\n"
                for info in analysis_data.found_information:
                    full_analysis += f"**Source:** {info.source}\n"
                    for quote in info.quotes:
                        full_analysis += f'- "{quote}"\n'
                    full_analysis += "\n"
                full_analysis += (
                    f"### Analyse actuelle:\n{analysis_data.current_analysis}\n\n"
                )
                all_analysis.append(full_analysis)

                if not analysis_data.need_more_info:
                    break
            except json.JSONDecodeError:
                st.error("Erreur lors du traitement de l'analyse du contenu")
                return None

        iteration += 1
        if iteration != max_iterations:
            with st.expander(
                f"🔁 Nouvelle itération - Itération {iteration}", expanded=True
            ):
                if analysis_data.need_more_info:
                    st.markdown(
                        f"**Informations supplémentaires nécessaires:** {analysis_data.additional_info_needed}"
                    )
                st.markdown(f"**Nouvelle question:** {question}")

    return analysis_data


# Main interface
st.markdown('<p class="big-font">Posez votre question :</p>', unsafe_allow_html=True)
question = st.text_area("", "Est-ce que LFI est d'extrême gauche ?")

if st.button("Rechercher", type="primary"):
    final_answer = process_question(question)
    st.success("Recherche terminée !")

    # st.markdown("### Réponse finale")
    # for info in final_answer.found_information:
    #    st.markdown(f"**Source:** {info.source}")
    #    for quote in info.quotes:
    #        st.markdown(f'- "{quote}"')

    # st.markdown(f"### Analyse actuelle:")
    # st.markdown(final_answer.current_analysis)

# Footer
st.markdown("---")
st.markdown(
    "Prototype développé par @superi4n en ~7 heures (toujours améliorable de zinzin)"
)
