"""Compara a contagem automática com a contagem manual de dois observadores.

Arquivo manual (CSV), uma linha por vídeo:
    video,observador1,observador2
    mindu_01,23,25
    mindu_02,11,10

O nome do vídeo é o nome do arquivo sem extensão, igual ao da pasta criada pelo
07_contar_video.py em resultados/<video>/.

Exemplo:
    python scripts/08_avaliar_contagem.py --manual dados/contagem_manual.csv --resultados resultados
"""
import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manual", required=True)
    ap.add_argument("--resultados", default="resultados")
    ap.add_argument("--saida", default="resultados/avaliacao_contagem.csv")
    a = ap.parse_args()

    linhas = []
    with open(a.manual, encoding="utf-8") as f:
        for ln in csv.DictReader(f):
            v = ln["video"].strip()
            o1, o2 = float(ln["observador1"]), float(ln["observador2"])
            arq = Path(a.resultados) / v / "contagens.csv"
            if not arq.exists():
                print(f"[aviso] sem resultado automático para {v} ({arq})")
                continue
            with arq.open(encoding="utf-8") as g:
                auto = sum(1 for _ in csv.DictReader(g))
            ref = (o1 + o2) / 2
            erro = auto - ref
            linhas.append({"video": v, "observador1": o1, "observador2": o2, "referencia": ref, "automatica": auto,
                           "erro": erro, "erro_abs": abs(erro), "erro_pct": (abs(erro) / ref * 100) if ref else None,
                           "dif_observadores": abs(o1 - o2)})
    if not linhas:
        raise SystemExit("Nenhum vídeo para comparar.")

    n = len(linhas)
    mae = sum(l["erro_abs"] for l in linhas) / n
    pcts = [l["erro_pct"] for l in linhas if l["erro_pct"] is not None]
    mape = sum(pcts) / len(pcts) if pcts else float("nan")
    vies = sum(l["erro"] for l in linhas) / n
    dif_obs = sum(l["dif_observadores"] for l in linhas) / n

    saida = Path(a.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    with saida.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)

    print(f"{'vídeo':20} {'manual':>7} {'auto':>6} {'erro':>6}")
    for l in linhas:
        print(f"{l['video']:20} {l['referencia']:7.1f} {l['automatica']:6d} {l['erro']:+6.1f}")
    print(f"\nErro absoluto médio (MAE): {mae:.2f} objetos por vídeo")
    print(f"Erro percentual médio:     {mape:.1f}%")
    print(f"Viés (auto - manual):      {vies:+.2f}  (positivo = sistema conta a mais)")
    print(f"Diferença média entre os observadores: {dif_obs:.2f}")
    print(f"Detalhes em {saida}")


if __name__ == "__main__":
    main()
