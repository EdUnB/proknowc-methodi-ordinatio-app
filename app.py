import streamlit as st
import pandas as pd

st.set_page_config(page_title="ProKnow-C + Methodi Ordinatio", layout="wide")

st.title("ProKnow-C & Methodi Ordinatio")
st.write("Aplicação para análise bibliométrica usando Scopus e Scimago")

st.header("1. Upload dos arquivos")

scopus_file = st.file_uploader("Upload do CSV do Scopus", type=["csv"])
scimago_file = st.file_uploader("Upload do CSV do Scimago", type=["csv"])

if scopus_file and scimago_file:
    st.success("Arquivos carregados com sucesso!")

    df_scopus = pd.read_csv(scopus_file)
    df_scimago = pd.read_csv(scimago_file, sep=";", encoding="utf-8-sig")

    st.header("2. Prévia dos dados")

    st.subheader("Scopus")
    st.dataframe(df_scopus.head())

    st.subheader("Scimago")
    st.dataframe(df_scimago.head())

    st.info("Pipeline completo será executado nas próximas versões.")
