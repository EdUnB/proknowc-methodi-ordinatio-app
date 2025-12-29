import re
import time
import io
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import numpy as np
import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="ProKnow C and Methodi Ordinatio", layout="wide")
st.title("ProKnow C and Methodi Ordinatio")
st.caption("Scopus CSV como entrada principal, integração opcional com Scimago SJR, recuperação de ISSN via DOI e etapa manual quando necessário.")


def normalize_issn(value):
    if value is None or pd.isna(value):
        return ""
    s = str(value).strip()
    s = re.sub(r"\.0$", "", s)
    digits = re.sub(r"[^0-9Xx]", "", s).upper()
    if digits == "":
        return ""
    blocks = re.findall(r"\d{8}", digits)
    if len(blocks) >= 1:
        digits = blocks[0]
    if len(digits) == 7 and digits.isdigit():
        digits = "0" + digits
    if len(digits) == 8:
        return f"{digits[:4]}-{digits[4:]}"
    return ""


def extract_all_issn_norm(value):
    if value is None or pd.isna(value):
        return []
    blocks = re.findall(r"\d{8}", str(value))
    out = [f"{b[:4]}-{b[4:]}" for b in blocks]
    seen = []
    for x in out:
        if x not in seen:
            seen.append(x)
    return seen


def parse_sjr_float(value):
    if value is None or pd.isna(value):
        return pd.NA
    s = str(value).strip()
    s = re.sub(r"[^0-9,\.]", "", s)
    if s == "":
        return pd.NA
    s = s.replace(".", "")
    s = s.replace(",", ".")
    return pd.to_numeric(s, errors="coerce")


def score_issn_series(series):
    s = series.dropna().astype(str).head(2000)
    if len(s) == 0:
        return 0.0
    ok = s.apply(lambda x: bool(re.search(r"\d{8}", x)))
    return float(ok.mean())


def score_quartile_series(series):
    s = series.dropna().astype(str).head(2000)
    if len(s) == 0:
        return 0.0
    ok = s.apply(lambda x: str(x).strip() in ["Q1", "Q2", "Q3", "Q4"])
    return float(ok.mean())


def score_sjr_series(series):
    s = series.dropna().astype(str).head(2000)
    if len(s) == 0:
        return 0.0
    ok = s.apply(lambda x: bool(re.match(r"^\s*\d{1,3}(?:\.\d{3})*,\d+\s*$|^\s*\d+,\d+\s*$", x)))
    return float(ok.mean())


def crossref_fetch_issn_from_doi(doi, contact_email=""):
    doi = str(doi).strip()
    if doi == "" or doi.lower() == "nan":
        return ""
    url = f"https://api.crossref.org/works/{doi}"
    ua = "ProKnowC-App/1.0"
    if contact_email.strip() != "":
        ua = f"ProKnowC-App/1.0 (contact: {contact_email.strip()})"
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": ua})
        if r.status_code != 200:
            return ""
        data = r.json()
    except Exception:
        return ""
    msg = data.get("message", {}) or {}
    issn_list = msg.get("ISSN", []) or []
    for it in issn_list:
        digits = re.sub(r"\D", "", str(it))
        if len(digits) == 8:
            return f"{digits[:4]}-{digits[4:]}"
    return ""


@st.cache_data(show_spinner=False)
def read_scopus(scopus_file):
    df_raw = pd.read_csv(scopus_file, dtype=str, keep_default_na=True)
    for col in ["Title", "Year", "Source title", "Cited by", "DOI", "ISSN"]:
        if col not in df_raw.columns:
            df_raw[col] = pd.NA

    df = pd.DataFrame({
        "title": df_raw["Title"].astype("string"),
        "year": pd.to_numeric(df_raw["Year"], errors="coerce").astype("Int64"),
        "journal": df_raw["Source title"].astype("string"),
        "cited_by": pd.to_numeric(df_raw["Cited by"], errors="coerce").fillna(0).astype(int),
        "doi": df_raw["DOI"].astype("string"),
        "issn": df_raw["ISSN"].astype("string"),
    })
    df["issn_norm"] = df["issn"].apply(normalize_issn)
    df["is_book_chapter"] = False
    return df


@st.cache_data(show_spinner=False)
def read_scimago(scimago_file):
    df_raw = pd.read_csv(
        scimago_file,
        sep=";",
        encoding="utf-8-sig",
        dtype=str,
        engine="python",
        on_bad_lines="skip"
    )

    cols = list(df_raw.columns)

    issn_scores = [(c, score_issn_series(df_raw[c])) for c in cols]
    issn_col, issn_score = sorted(issn_scores, key=lambda x: x[1], reverse=True)[0]

    sjr_scores = [(c, score_sjr_series(df_raw[c])) for c in cols]
    sjr_col, sjr_score = sorted(sjr_scores, key=lambda x: x[1], reverse=True)[0]

    quart_scores = [(c, score_quartile_series(df_raw[c])) for c in cols]
    quart_col, quart_score = sorted(quart_scores, key=lambda x: x[1], reverse=True)[0]

    df = pd.DataFrame({
        "issn_raw": df_raw[issn_col],
        "sjr": df_raw[sjr_col].apply(parse_sjr_float),
        "best_quartile": df_raw[quart_col].astype(str).str.strip() if quart_score >= 0.2 else pd.Series([pd.NA] * len(df_raw)),
    })

    df["issn_norm"] = df["issn_raw"].apply(extract_all_issn_norm)
    df = df.explode("issn_norm")
    df = df[df["issn_norm"].notna() & (df["issn_norm"] != "")]
    df_merge = df[["issn_norm", "sjr", "best_quartile"]].drop_duplicates(subset=["issn_norm"]).copy()

    meta = {
        "rows": int(len(df_raw)),
        "issn_col": issn_col,
        "issn_score": float(issn_score),
        "sjr_col": sjr_col,
        "sjr_score": float(sjr_score),
        "quart_col": quart_col,
        "quart_score": float(quart_score),
        "merge_keys": int(len(df_merge)),
    }
    return df_merge, meta


def make_key(df):
    j = df["journal"].fillna("").astype(str).str.strip().str.lower()
    t = df["title"].fillna("").astype(str).str.strip().str.lower()
    y = df["year"].astype("Int64").astype(str)
    return j + "||" + t + "||" + y


def compute_methodi_ordinatio(df_final, Yc, alpha):
    tabela = pd.DataFrame({
        "paper_title": df_final["title"].astype("string"),
        "journal": df_final["journal"].astype("string"),
        "year": pd.to_numeric(df_final["year"], errors="coerce").astype("Int64"),
        "citations": pd.to_numeric(df_final["cited_by"], errors="coerce").fillna(0).astype(int),
        "impact_factor_sjr": pd.to_numeric(df_final["sjr"], errors="coerce"),
        "sjr_quartile": df_final["best_quartile"].astype("string"),
        "issn_norm": df_final["issn_norm"].astype("string"),
        "doi": df_final["doi"].astype("string"),
        "is_book_chapter": df_final["is_book_chapter"].astype(bool),
    })

    tabela = tabela.dropna(subset=["paper_title", "year"])
    tabela = tabela[tabela["paper_title"].astype(str).str.strip() != ""]
    tabela["year"] = tabela["year"].astype(int)

    tabela["if_used"] = tabela["impact_factor_sjr"].fillna(0)
    tabela["recency_term"] = alpha * (10 - (Yc - tabela["year"]))
    tabela["inordinatio"] = tabela["recency_term"] + tabela["if_used"] + tabela["citations"]

    tabela = tabela.sort_values(
        by=["inordinatio", "citations", "impact_factor_sjr", "year"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    tabela.insert(0, "rank", tabela.index + 1)
    return tabela

def fig_to_png_bytes(fig, dpi=200):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()

def safe_journal_label(s):
    s = str(s)
    s = s.replace("/", "_").replace("\\", "_").replace(":", "_")
    return s[:80]

st.sidebar.header("Arquivos")
scopus_file = st.sidebar.file_uploader("Scopus CSV", type=["csv"])
scimago_file = st.sidebar.file_uploader("Scimago CSV", type=["csv"])

st.sidebar.header("Parâmetros")
Yc = st.sidebar.number_input("Ano de referência, Yc", min_value=1900, max_value=2100, value=2025, step=1)
alpha = st.sidebar.number_input("Alpha", min_value=0, max_value=100, value=10, step=1)

st.sidebar.header("Crossref")
contact_email = st.sidebar.text_input("Email para User Agent, opcional", value="")

st.sidebar.header("Ações")
run_auto = st.sidebar.button("Rodar etapa automática")
compute_final = st.sidebar.button("Calcular resultados finais")


if "state" not in st.session_state:
    st.session_state.state = {}


if run_auto:
    if scopus_file is None:
        st.error("Envie o CSV do Scopus.")
    else:
        with st.spinner("Lendo Scopus"):
            df_scopus = read_scopus(scopus_file)

        df_sjr_merge = None
        sjr_meta = None
        if scimago_file is not None:
            with st.spinner("Lendo Scimago e preparando tabela de merge"):
                df_sjr_merge, sjr_meta = read_scimago(scimago_file)

        with st.spinner("Recuperando ISSN via DOI, apenas para casos sem ISSN"):
            df_scopus_enriched = df_scopus.copy()
            mask = (df_scopus_enriched["issn_norm"] == "") & df_scopus_enriched["doi"].notna() & (df_scopus_enriched["doi"].astype(str).str.strip() != "")
            idxs = list(df_scopus_enriched.loc[mask].index)

            recovered = 0
            for i in idxs:
                doi = df_scopus_enriched.at[i, "doi"]
                issn_val = crossref_fetch_issn_from_doi(doi, contact_email=contact_email)
                if issn_val != "":
                    df_scopus_enriched.at[i, "issn_norm"] = issn_val
                    recovered += 1
                time.sleep(0.15)

        st.session_state.state["df_scopus_enriched"] = df_scopus_enriched
        st.session_state.state["df_sjr_merge"] = df_sjr_merge
        st.session_state.state["sjr_meta"] = sjr_meta
        st.session_state.state["recovered"] = int(recovered)
        st.success("Etapa automática concluída.")


if "df_scopus_enriched" in st.session_state.state:
    df_scopus_enriched = st.session_state.state["df_scopus_enriched"]
    df_sjr_merge = st.session_state.state.get("df_sjr_merge", None)
    sjr_meta = st.session_state.state.get("sjr_meta", None)
    recovered = st.session_state.state.get("recovered", 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros Scopus", int(len(df_scopus_enriched)))
    c2.metric("ISSN preenchidos", int((df_scopus_enriched["issn_norm"] != "").sum()))
    c3.metric("ISSN faltando", int((df_scopus_enriched["issn_norm"] == "").sum()))
    c4.metric("Recuperados via DOI", int(recovered))

    if sjr_meta is not None:
        with st.expander("Diagnóstico de leitura do Scimago"):
            st.write(sjr_meta)

    st.subheader("Registros ainda sem ISSN")
    missing = df_scopus_enriched.loc[df_scopus_enriched["issn_norm"] == "", ["journal", "title", "year", "doi"]].copy()
    st.dataframe(missing, use_container_width=True, height=220)

    st.subheader("Etapa manual de ISSN")
    st.write("Baixe o modelo, preencha issn_norm_manual no formato ####-####. Para capítulo de livro, use notes com BOOK_CHAPTER.")

    manual = missing.copy()

    # Coluna principal para o usuário preencher
    manual["ISSN_PREENCHER_AQUI"] = ""

    # Ajuda visual
    manual["FORMATO_ESPERADO"] = "####-####"
    manual["EXEMPLO"] = "0360-5442"
    manual["INSTRUCAO"] = "Preencha ISSN_PREENCHER_AQUI. Se for capitulo de livro, escreva BOOK_CHAPTER em TIPO_ITEM."

    # Campo para marcar excecoes
    manual["TIPO_ITEM"] = ""

    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
        manual.to_excel(writer, index=False, sheet_name="manual_issn")

    st.download_button(
        label="Baixar modelo para preenchimento manual",
        data=xlsx_buf.getvalue(),
        file_name="issn_manual_fill_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    filled = st.file_uploader("Enviar modelo preenchido", type=["xlsx"])

    if filled is not None:
        manual_filled = pd.read_excel(filled, dtype=str)
        needed = {"journal", "title", "year", "doi", "ISSN_PREENCHER_AQUI", "TIPO_ITEM"}
        if not needed.issubset(set(manual_filled.columns)):
            st.error("Modelo não contém todas as colunas necessárias.")
        else:
            manual_filled["year"] = pd.to_numeric(manual_filled["year"], errors="coerce").astype("Int64")
            manual_filled["ISSN_PREENCHER_AQUI"] = manual_filled["ISSN_PREENCHER_AQUI"].apply(normalize_issn)
            manual_filled["TIPO_ITEM"] = manual_filled["TIPO_ITEM"].fillna("").astype(str).str.strip().str.upper()

            issn_map = dict(zip(manual_filled["match_key"], manual_filled["ISSN_PREENCHER_AQUI"]))
            tipo_map = dict(zip(manual_filled["match_key"], manual_filled["TIPO_ITEM"]))
            
            df_updated["issn_norm_manual"] = df_updated["match_key"].map(issn_map).fillna("")
            df_updated["manual_tipo_item"] = df_updated["match_key"].map(tipo_map).fillna("")
            
            mask_apply = (df_updated["issn_norm"] == "") & (df_updated["issn_norm_manual"] != "")
            df_updated.loc[mask_apply, "issn_norm"] = df_updated.loc[mask_apply, "issn_norm_manual"]
            
            df_updated["is_book_chapter"] = df_updated["manual_tipo_item"].str.contains("BOOK_CHAPTER", na=False)

            st.session_state.state["df_scopus_enriched"] = df_updated
            st.success(f"ISSN aplicados manualmente: {int(mask_apply.sum())}")


if compute_final:
    if "df_scopus_enriched" not in st.session_state.state:
        st.error("Rode a etapa automática primeiro.")
    else:
        df_scopus_enriched = st.session_state.state["df_scopus_enriched"]
        df_sjr_merge = st.session_state.state.get("df_sjr_merge", None)

        if df_sjr_merge is None:
            df_final = df_scopus_enriched.copy()
            df_final["sjr"] = pd.NA
            df_final["best_quartile"] = pd.NA
        else:
            df_final = df_scopus_enriched.merge(df_sjr_merge, on="issn_norm", how="left")

        mask_book = df_final["is_book_chapter"] == True
        df_final.loc[mask_book, "sjr"] = pd.NA
        df_final.loc[mask_book, "best_quartile"] = pd.NA

        tabela = compute_methodi_ordinatio(df_final, int(Yc), int(alpha))

        st.session_state.state["df_final"] = df_final
        st.session_state.state["tabela"] = tabela

        st.success("Resultados calculados.")


if "tabela" in st.session_state.state:
    tabela = st.session_state.state["tabela"]
    df_final = st.session_state.state["df_final"]

    st.subheader("Tabela ranqueada")
    st.dataframe(tabela, use_container_width=True, height=420)

   st.subheader("Gráficos e exportação")

    # B1.1 Publicações por ano
    st.markdown("### B1.1 Publicações por ano")
    by_year = (
        tabela.groupby("year", dropna=True)
        .size()
        .reset_index(name="count")
        .sort_values("year")
    )
    fig1, ax1 = plt.subplots()
    ax1.bar(by_year["year"].astype(int), by_year["count"].astype(int))
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of publications")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    st.pyplot(fig1, clear_figure=True)
    png1 = fig_to_png_bytes(fig1)
    st.download_button(
        "Download PNG (B1.1)",
        data=png1,
        file_name="B1_1_publications_by_year.png",
        mime="image/png",
    )
    
    # B1.2 Top periódicos por número de artigos
    st.markdown("### B1.2 Top periódicos por número de artigos")
    top_n = st.slider("Top N periódicos", min_value=5, max_value=30, value=15, step=1)
    by_j = (
        tabela["journal"].fillna("Unknown")
        .astype(str)
        .value_counts()
        .head(top_n)
        .reset_index()
    )
    by_j.columns = ["journal", "count"]
    by_j = by_j.sort_values("count", ascending=True)
    
    fig2, ax2 = plt.subplots()
    ax2.barh(by_j["journal"], by_j["count"].astype(int))
    ax2.set_xlabel("Number of publications")
    ax2.set_ylabel("Journal")
    st.pyplot(fig2, clear_figure=True)
    png2 = fig_to_png_bytes(fig2)
    st.download_button(
        "Download PNG (B1.2)",
        data=png2,
        file_name=f"B1_2_top_{top_n}_journals.png",
        mime="image/png",
    )
    
    # B1.3 Distribuição do InOrdinatio
    st.markdown("### B1.3 Distribuição do InOrdinatio")
    bins = st.slider("Número de bins", min_value=5, max_value=80, value=30, step=1)
    
    ord_vals = pd.to_numeric(tabela["inordinatio"], errors="coerce").dropna()
    fig3, ax3 = plt.subplots()
    ax3.hist(ord_vals.values, bins=int(bins))
    ax3.set_xlabel("InOrdinatio")
    ax3.set_ylabel("Frequency")
    st.pyplot(fig3, clear_figure=True)
    png3 = fig_to_png_bytes(fig3)
    st.download_button(
        "Download PNG (B1.3)",
        data=png3,
        file_name=f"B1_3_inordinatio_distribution_bins_{bins}.png",
        mime="image/png",
    )
    
    # Complementares rápidos, exportáveis também
    st.markdown("### Complementares")
    
    st.markdown("#### Top 15 por citações")
    top_cit = tabela.sort_values("citations", ascending=False).head(15)[["paper_title", "citations"]].copy()
    top_cit["paper_title"] = top_cit["paper_title"].astype(str).str.slice(0, 80)
    
    fig4, ax4 = plt.subplots()
    ax4.barh(top_cit["paper_title"][::-1], top_cit["citations"][::-1].astype(int))
    ax4.set_xlabel("Citations")
    ax4.set_ylabel("Paper title")
    st.pyplot(fig4, clear_figure=True)
    png4 = fig_to_png_bytes(fig4)
    st.download_button(
        "Download PNG (Top citations)",
        data=png4,
        file_name="top_15_citations.png",
        mime="image/png",
    )
    
    st.markdown("#### Top 15 por InOrdinatio")
    top_ord = tabela.sort_values("inordinatio", ascending=False).head(15)[["paper_title", "inordinatio"]].copy()
    top_ord["paper_title"] = top_ord["paper_title"].astype(str).str.slice(0, 80)
    
    fig5, ax5 = plt.subplots()
    ax5.barh(top_ord["paper_title"][::-1], pd.to_numeric(top_ord["inordinatio"], errors="coerce")[::-1])
    ax5.set_xlabel("InOrdinatio")
    ax5.set_ylabel("Paper title")
    st.pyplot(fig5, clear_figure=True)
    png5 = fig_to_png_bytes(fig5)
    st.download_button(
        "Download PNG (Top InOrdinatio)",
        data=png5,
        file_name="top_15_inordinatio.png",
        mime="image/png",
    )

    st.subheader("Downloads")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    csv_bytes = tabela.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
        tabela.to_excel(writer, index=False, sheet_name="methodi_ordinatio")

    st.download_button(
        "Baixar CSV final",
        data=csv_bytes,
        file_name=f"proknowc_methodi_ordinatio_FINAL_{timestamp}.csv",
        mime="text/csv",
    )
    st.download_button(
        "Baixar XLSX final",
        data=xlsx_buf.getvalue(),
        file_name=f"proknowc_methodi_ordinatio_FINAL_{timestamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.subheader("Diagnóstico")
    total = int(len(df_final))
    with_sjr = int(df_final["sjr"].notna().sum())
    without_sjr = int(df_final["sjr"].isna().sum())
    book_chapter = int(df_final["is_book_chapter"].sum())

    st.write({
        "total_itens": total,
        "com_sjr": with_sjr,
        "sem_sjr": without_sjr,
        "book_chapter": book_chapter,
    })


