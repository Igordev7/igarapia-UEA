"""Avalia um ou mais modelos no conjunto de TESTE local e gera uma tabela comparativa.

Exemplo (comparar os três cenários):
    python scripts/06_avaliar_deteccao.py --dados cenarios/E3.yaml \
        --modelos runs/E1_yolo11n/weights/best.pt runs/E2_yolo11n/weights/best.pt runs/E3_yolo11n/weights/best.pt \
        --nomes E1 E2 E3 --saida resultados/deteccao.csv

Todos os cenários usam o mesmo teste (dados/local/test), então qualquer um dos
arquivos de cenário serve em --dados.
Métricas: precisão (P), revocação (R), mAP50 e mAP50-95, no total e por classe.
"""
import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


def avaliar(caminho, nome, dados, imgsz, dispositivo):
    m = YOLO(caminho)
    r = m.val(data=dados, split="test", imgsz=imgsz, device=dispositivo, plots=True,
              project=str(Path("runs").resolve()), name=f"teste_{nome}", exist_ok=True, verbose=False)
    b = r.box
    linhas = [{"modelo": nome, "classe": "todas", "P": b.mp, "R": b.mr, "mAP50": b.map50, "mAP50-95": b.map}]
    for i, c in enumerate(b.ap_class_index):
        linhas.append({"modelo": nome, "classe": r.names[int(c)], "P": b.p[i], "R": b.r[i], "mAP50": b.ap50[i], "mAP50-95": b.ap[i]})
    vel = r.speed.get("inference")
    for ln in linhas:
        ln["ms_por_imagem"] = vel
    return linhas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dados", required=True)
    ap.add_argument("--modelos", nargs="+", required=True)
    ap.add_argument("--nomes", nargs="+", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--dispositivo", default=None)
    ap.add_argument("--saida", default="resultados/deteccao.csv")
    a = ap.parse_args()

    nomes = a.nomes or [Path(m).parent.parent.name for m in a.modelos]
    if len(nomes) != len(a.modelos):
        raise SystemExit("--nomes precisa ter um nome para cada modelo")
    todas = []
    for m, n in zip(a.modelos, nomes):
        todas += avaliar(m, n, a.dados, a.imgsz, a.dispositivo)

    saida = Path(a.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    campos = ["modelo", "classe", "P", "R", "mAP50", "mAP50-95", "ms_por_imagem"]
    with saida.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        for ln in todas:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in ln.items()})

    print(f"\n{'modelo':8} {'classe':24} {'P':>6} {'R':>6} {'mAP50':>7} {'mAP50-95':>9}")
    for ln in todas:
        print(f"{ln['modelo']:8} {ln['classe']:24} {ln['P']:6.3f} {ln['R']:6.3f} {ln['mAP50']:7.3f} {ln['mAP50-95']:9.3f}")
    print(f"\nTabela salva em {saida}")


if __name__ == "__main__":
    main()
