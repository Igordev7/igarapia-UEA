"""Calcula MOTA e IDF1 comparando o rastreamento do sistema com trechos anotados à mão.

Gabarito: anote o trecho no CVAT com identificadores (track) e exporte no formato
"MOT 1.1"; use o arquivo gt/gt.txt. Previsão: rode o 07_contar_video.py com
--salvar-mot e use resultados/<video>/rastreamento_mot.txt.

Exemplo:
    python scripts/09_avaliar_rastreamento.py --gt anotacoes_mot/mindu_03/gt/gt.txt \
        --pred resultados/mindu_03/rastreamento_mot.txt
"""
import argparse

import numpy as np

# compatibilidade do motmetrics com o NumPy 2
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

import motmetrics as mm  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--iou", type=float, default=0.5, help="IoU mínimo para considerar a mesma caixa")
    a = ap.parse_args()

    gt = mm.io.loadtxt(a.gt, fmt="mot15-2D", min_confidence=0)
    pred = mm.io.loadtxt(a.pred, fmt="mot15-2D")
    acc = mm.utils.compare_to_groundtruth(gt, pred, "iou", distth=1 - a.iou)
    mh = mm.metrics.create()
    res = mh.compute(acc, metrics=["mota", "motp", "idf1", "precision", "recall", "num_switches",
                                   "num_false_positives", "num_misses", "num_unique_objects"], name="trecho")
    print(mm.io.render_summary(res, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))


if __name__ == "__main__":
    main()
