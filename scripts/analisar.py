"""Analisa imagens ou vídeos de igarapés: detecta, conta os resíduos por tipo e gera relatório.

Exemplos:
    python scripts/analisar.py --fonte video_igarape.mp4 --local "Igarapé do Mindu"
    python scripts/analisar.py --fonte pasta_de_fotos/ --local "Igarapé do Quarenta" --conf 0.3
"""
import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from igarapia.nucleo import (analisar_imagem, analisar_video, carregar_modelo, listar_fontes,  # noqa: E402
                             salvar_resultado, tipo_da_fonte)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fonte", required=True, help="imagem, vídeo ou pasta")
    ap.add_argument("--modelo", default=str(RAIZ / "modelos" / "igarapia_best.pt"))
    ap.add_argument("--conf", type=float, default=0.25, help="confiança mínima para contar uma detecção")
    ap.add_argument("--local", default="não informado", help="nome do igarapé ou trecho")
    ap.add_argument("--passo", type=int, default=2, help="em vídeos, analisa 1 a cada N quadros")
    ap.add_argument("--saida", default=str(RAIZ / "resultados"))
    ap.add_argument("--imgsz", type=int, default=640, help="tamanho da imagem na análise; 960 ajuda com objetos pequenos")
    ap.add_argument("--por-classe", action="store_true", help="permite caixas de classes diferentes sobre o mesmo objeto")
    ap.add_argument("--tta", action="store_true", help="análise reforçada em imagens (mais lenta)")
    a = ap.parse_args()

    modelo = carregar_modelo(a.modelo)
    for fonte in listar_fontes(a.fonte):
        if tipo_da_fonte(fonte) == "video":
            res = analisar_video(modelo, fonte, a.saida, a.conf, a.local, a.passo, imgsz=a.imgsz, agnostico=not a.por_classe)
        else:
            res = analisar_imagem(modelo, fonte, a.saida, a.conf, a.local, imgsz=a.imgsz, agnostico=not a.por_classe, tta=a.tta)
        resumo = salvar_resultado(res, a.saida)
        print(f"\n{res.fonte} ({res.tipo}) – {resumo['total_itens']} itens")
        for classe, qtd in resumo["itens_por_classe"].items():
            print(f"  {classe:<24}{qtd:>5}")
        print(f"  arquivo anotado: {res.saida_anotada}")
    print(f"\nResultados e histórico em: {a.saida}")


if __name__ == "__main__":
    main()
