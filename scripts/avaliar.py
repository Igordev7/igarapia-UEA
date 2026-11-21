"""Avalia o modelo no conjunto de teste e grava as métricas por classe.

Uso:
    python scripts/avaliar.py --modelo modelos/igarapia_best.pt --dados dados/igarapia/data.yaml
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", default=str(RAIZ / "modelos" / "igarapia_best.pt"))
    ap.add_argument("--dados", required=True)
    ap.add_argument("--divisao", default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--saida", default=str(RAIZ / "resultados"))
    a = ap.parse_args()

    m = YOLO(a.modelo).val(data=a.dados, split=a.divisao, imgsz=a.imgsz, plots=True,
                           project=str(RAIZ / "resultados"), name=f"avaliacao_{a.divisao}", exist_ok=True)
    nomes = m.names
    por_classe = {}
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        por_classe[nomes[int(c)]] = {"precisao": round(p, 3), "revocacao": round(r, 3),
                                     "mAP50": round(ap50, 3), "mAP50_95": round(ap, 3)}
    geral = {"precisao": round(m.box.mp, 3), "revocacao": round(m.box.mr, 3),
             "mAP50": round(m.box.map50, 3), "mAP50_95": round(m.box.map, 3)}
    Path(a.saida).mkdir(parents=True, exist_ok=True)
    out = Path(a.saida) / f"metricas_{a.divisao}.json"
    json.dump({"geral": geral, "por_classe": por_classe}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n{'classe':<24}{'precisão':>10}{'revocação':>11}{'mAP50':>8}{'mAP50-95':>10}")
    for c, v in por_classe.items():
        print(f"{c:<24}{v['precisao']:>10}{v['revocacao']:>11}{v['mAP50']:>8}{v['mAP50_95']:>10}")
    print(f"{'GERAL':<24}{geral['precisao']:>10}{geral['revocacao']:>11}{geral['mAP50']:>8}{geral['mAP50_95']:>10}")
    print(f"\nMétricas gravadas em: {out}")


if __name__ == "__main__":
    main()
