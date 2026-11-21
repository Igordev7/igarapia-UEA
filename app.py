"""Painel do IgarapIA (Streamlit).

Execute com:  streamlit run app.py
"""
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from igarapia.nucleo import analisar_imagem, analisar_video, carregar_modelo, salvar_resultado, tipo_da_fonte

RAIZ = Path(__file__).resolve().parent
PASTA_RESULTADOS = RAIZ / "resultados"
HISTORICO = PASTA_RESULTADOS / "historico.csv"

st.set_page_config(page_title="IgarapIA", layout="wide")
st.title("IgarapIA – detecção e contagem de resíduos em igarapés")


@st.cache_resource
def modelo_em_cache(caminho):
    return carregar_modelo(caminho)


with st.sidebar:
    st.header("Configuração")
    caminho_modelo = st.text_input("Modelo", "modelos/igarapia_best.pt")
    if not Path(caminho_modelo).is_absolute():
        caminho_modelo = str(RAIZ / caminho_modelo)
    conf = st.slider("Confiança mínima", 0.05, 0.90, 0.25, 0.05)
    local = st.text_input("Local (igarapé ou trecho)", "Teste com imagem do TACO")
    passo = st.number_input("Vídeo: analisar 1 a cada N quadros", 1, 30, 3)

aba_analise, aba_historico = st.tabs(["Analisar imagem ou vídeo", "Histórico"])

with aba_analise:
    arquivo = st.file_uploader("Envie uma foto ou vídeo do igarapé",
                               type=["jpg", "jpeg", "png", "webp", "mp4", "avi", "mov", "mkv"])
    if arquivo and st.button("Analisar", type="primary"):
        if not Path(caminho_modelo).exists():
            st.error("Modelo não encontrado. Treine com scripts/treinar.py ou informe outro caminho.")
            st.stop()
        modelo = modelo_em_cache(caminho_modelo)
        pasta_tmp = Path(tempfile.mkdtemp())
        caminho = pasta_tmp / arquivo.name
        caminho.write_bytes(arquivo.getbuffer())
        if tipo_da_fonte(str(caminho)) == "video":
            barra = st.progress(0.0, "Processando vídeo...")
            res = analisar_video(modelo, str(caminho), str(PASTA_RESULTADOS), conf, local, int(passo),
                                 ao_progredir=lambda p: barra.progress(p, "Processando vídeo..."))
            barra.empty()
        else:
            res = analisar_imagem(modelo, str(caminho), str(PASTA_RESULTADOS), conf, local)
        resumo = salvar_resultado(res, str(PASTA_RESULTADOS), str(HISTORICO))

        col1, col2 = st.columns([3, 2])
        with col1:
            if res.tipo == "imagem":
                st.image(res.saida_anotada, caption="Resíduos detectados", use_container_width=True)
            elif res.saida_anotada:
                st.info("Vídeo anotado gerado. Use o botão abaixo para baixá-lo.")
                st.download_button("Baixar vídeo anotado", open(res.saida_anotada, "rb"),
                                   file_name=Path(res.saida_anotada).name)
        with col2:
            st.metric("Total de itens", resumo["total_itens"])
            if res.contagem:
                df = pd.DataFrame(sorted(res.contagem.items(), key=lambda x: -x[1]), columns=["Classe", "Quantidade"])
                st.bar_chart(df, x="Classe", y="Quantidade")
                st.dataframe(df, hide_index=True, use_container_width=True)
            else:
                st.write("Nenhum resíduo detectado com a confiança escolhida.")
            det = pd.DataFrame(res.deteccoes)
            st.download_button("Baixar detecções (CSV)", det.to_csv(index=False).encode("utf-8"),
                               file_name=f"deteccoes_{Path(res.fonte).stem}.csv")

with aba_historico:
    if HISTORICO.exists():
        h = pd.read_csv(HISTORICO)
        h = h[h["classe"] != "nenhum_item"]
        locais = sorted(h["local"].dropna().unique())
        escolha = st.multiselect("Locais", locais, default=locais)
        h = h[h["local"].isin(escolha)]
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Itens por tipo de resíduo")
            st.bar_chart(h.groupby("classe")["quantidade"].sum())
        with c2:
            st.subheader("Itens por local")
            st.bar_chart(h.groupby("local")["quantidade"].sum())
        st.subheader("Registros")
        st.dataframe(h.sort_values("data_analise", ascending=False), hide_index=True, use_container_width=True)
        st.download_button("Baixar histórico (CSV)", h.to_csv(index=False).encode("utf-8"), file_name="historico.csv")
    else:
        st.write("Ainda não há análises registradas.")
