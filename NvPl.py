import os
import time
import logging
import toml
import streamlit as st
import google.generativeai as genai
import plotly.express as px
from streamlit_option_menu import option_menu
import pandas as pd
import spacy
from wordcloud import WordCloud
from collections import Counter
from streamlit_autorefresh import st_autorefresh
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from streamlit_extras.switch_page_button import switch_page


logging.basicConfig(level=logging.DEBUG)

    st.markdown(
    """
    <style>
    div.block-container {
        padding-left: 2rem;
        padding-right: 2rem;
    }
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    img {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    .centered {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    st_autorefresh(interval=5120000, key="data_refresh")

    # ================================
    # 🔥 CARREGAMENTO SEGURO DO SPACY
    # ================================
    @st.cache_resource
    def load_spacy_pt():
        try:
            return spacy.load("pt_core_news_sm"), "pt_core_news_sm"
        except:
            try:
                return spacy.load("pt_core_news_md"), "pt_core_news_md"
            except:
                try:
                    return spacy.load("pt_core_news_lg"), "pt_core_news_lg"
                except:
                    # fallback — funciona na nuvem sem modelo instalado
                    from spacy.lang.pt import Portuguese
                    nlp_blank = Portuguese()
                    if "sentencizer" not in nlp_blank.pipe_names:
                        nlp_blank.add_pipe("sentencizer")
                    return nlp_blank, "blank_pt"

    nlp, MODEL_SPACY = load_spacy_pt()
    st.info(f"Modelo spaCy carregado: **{MODEL_SPACY}**")

    # ================================
    # 🔥 CARREGAR DADOS
    # ================================
    @st.cache_data(ttl=30)
    def load_data(csv_url):
        df = pd.read_csv(csv_url)

        # renomeia a segunda coluna para "percepcao"
        if len(df.columns) > 1:
            original = df.columns[1]
            df.rename(columns={original: "percepcao"}, inplace=True)
            logging.debug(f"Coluna '{original}' renomeada para 'percepcao'.")

        return df

    csv_url = "https://docs.google.com/spreadsheets/d/1dsAaDSCpLYts8Y9P6Jbd62yLaHTjvUN_B3H8XBH-JbQ/export?format=csv&id=1dsAaDSCpLYts8Y9P6Jbd62yLaHTjvUN_B3H8XBH-JbQ&gid=1585034273"

    try:
        data = load_data(csv_url)
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        st.stop()

    # ================================
    # 🔥 PROCESSAMENTO DE TEXTO
    # ================================
    def process_texts(texts):
        doc = nlp(" ".join(texts))
        tokens = []

        for token in doc:
            if not token.is_alpha:
                continue
            if getattr(token, "is_stop", False):
                continue

            lemma = token.lemma_.lower() if hasattr(token, "lemma_") else token.text.lower()

            if hasattr(token, "pos_") and token.pos_:
                if token.pos_ in ("VERB", "NOUN", "PROPN", "ADJ"):
                    tokens.append(lemma)
            else:
                if len(lemma) > 2:
                    tokens.append(lemma)

        return tokens

    exclude_words = [
        "ruim", "radiação", "cabos", "Poluição", "Acúmulo", "contaminavel",
        "Perigo", "sujeira", "Sistentabily", "Reversão", "Utópico",
        "Se o mundo comessase q descartar corretamente, o meio ambiente vai ter a oportunidade de se regenerar"
    ]

    tokens = None
    wordcloud_image = None
    freq_fig = None

    if "percepcao" in data.columns and not data["percepcao"].dropna().empty:
        texts = data["percepcao"].dropna().tolist()
        tokens = process_texts(texts)
        tokens = [t for t in tokens if t not in exclude_words]
    else:
        st.write("Coluna 'percepcao' não encontrada ou vazia.")

    # ================================
    # 🔥 NUVEM DE PALAVRAS
    # ================================
    def generate_wordcloud(tokens):
        freq = Counter(tokens)
        wc = WordCloud(
            width=600,
            height=600,
            background_color="white",
            colormap="viridis",
            max_words=100
        )
        wc.generate_from_frequencies(freq)
        return wc.to_array()

    def create_frequency_data(tokens):
        freq = Counter(tokens)
        df_freq = pd.DataFrame(freq.items(), columns=["palavra", "frequencia"])
        return df_freq.sort_values(by="frequencia", ascending=False).head(10)

    def create_frequency_chart(tokens):
        df_freq = create_frequency_data(tokens)
        fig = px.bar(
            df_freq,
            x="palavra",
            y="frequencia",
            text="frequencia",
            labels={"palavra": "Percepção", "frequencia": "Frequência"},
            color="frequencia",
            color_continuous_scale=[
                "#003300", "#006600", "#009933", "#33cc33", "#99ff99"
            ]
        )
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.update_layout(xaxis_tickangle=-45, margin=dict(t=40, b=40))
        return fig

    if tokens:
        wordcloud_image = generate_wordcloud(tokens)
        freq_fig = create_frequency_chart(tokens)

    # ================================
    # 🔥 EXIBIÇÃO
    # ================================
    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='centered'>", unsafe_allow_html=True)
            st.markdown("###### :bust_in_silhouette: Opiniões — E-lixo")
            if wordcloud_image is not None:
                st.image(wordcloud_image, use_container_width=True)
            else:
                st.write("Sem dados suficientes.")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='centered'>", unsafe_allow_html=True)
            st.markdown("###### :bust_in_silhouette: Contagem de palavras")
            if freq_fig is not None:
                st.plotly_chart(freq_fig, use_container_width=True)
            else:
                st.write("Sem dados suficientes.")
            st.markdown("</div>", unsafe_allow_html=True)

    # ================================
    # 🔥 DEPURAÇÃO
    # ================================
    st.markdown("---")
    with st.expander("Informações de Depuração"):
        st.write("##### Colunas do DataFrame:")
        st.write(data.columns.tolist())

        st.write("##### Primeiras 5 linhas:")
        st.dataframe(data.head())

        if tokens:
            st.write(f"Tokens extraídos: {len(tokens)}")
            st.write(tokens[:15])
        else:
            st.write("Nenhum token extraído.")
