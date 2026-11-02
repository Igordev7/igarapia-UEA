"""Gera dados de brinquedo para testar o fluxo inteiro sem vídeos reais.

Cria vídeos curtos com "objetos" descendo sobre um fundo cor de água, os rótulos
YOLO de cada quadro, uma contagem manual (gabarito exato), um arquivo MOT de
gabarito para o rastreamento e uma pequena base pública falsa em formato COCO.
Os números obtidos com esses dados NÃO têm valor científico: servem só para
verificar que os scripts funcionam na sua máquina.

    python tests/gerar_dados_sinteticos.py --saida demo
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

CORES = {0: (235, 235, 240), 1: (40, 40, 45), 2: (250, 250, 250)}  # garrafa, sacola, isopor (BGR)
TAM = {0: (34, 14), 1: (26, 22), 2: (22, 22)}  # meia-largura, meia-altura


def fundo(w, h, rng):
    base = np.zeros((h, w, 3), np.uint8)
    base[:] = (40, 70, 85)  # água escura
    ruido = rng.normal(0, 8, (h, w, 1)).astype(np.int16)
    return np.clip(base.astype(np.int16) + ruido, 0, 255).astype(np.uint8)


def desenha(img, cls, cx, cy):
    a, b = TAM[cls]
    if cls == 0:
        cv2.ellipse(img, (int(cx), int(cy)), (a, b), 0, 0, 360, CORES[cls], -1)
        cv2.rectangle(img, (int(cx + a - 4), int(cy - 5)), (int(cx + a + 6), int(cy + 5)), (60, 120, 220), -1)
    elif cls == 1:
        pts = np.array([[cx - a, cy - b], [cx + a, cy - b + 6], [cx + a - 4, cy + b], [cx - a + 5, cy + b - 3]], np.int32)
        cv2.fillPoly(img, [pts], CORES[cls])
    else:
        cv2.rectangle(img, (int(cx - a), int(cy - b)), (int(cx + a), int(cy + b)), CORES[cls], -1)
    extra = 6 if cls == 0 else 0
    return cx - a, cy - b, cx + a + extra, cy + b


def gerar_video(nome, pasta_v, pasta_r, mot_dir, w, h, fps, seg, rng):
    n = fps * seg
    objs = []
    for i in range(rng.integers(3, 7)):
        objs.append({"id": i + 1, "cls": int(rng.integers(0, 3)), "x": float(rng.uniform(40, w - 40)),
                     "y0": float(rng.uniform(-60, h * 0.3)), "v": float(rng.uniform(1.5, 3.0)), "t0": int(rng.integers(0, n // 2))})
    out = cv2.VideoWriter(str(pasta_v / f"{nome}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    (pasta_r / nome).mkdir(parents=True, exist_ok=True)
    gt = []
    for f in range(n):
        img = fundo(w, h, rng)
        linhas = []
        for o in objs:
            if f < o["t0"]:
                continue
            cy = o["y0"] + (f - o["t0"]) * o["v"]
            cx = o["x"] + 6 * np.sin((f + o["id"] * 7) / 15)
            x1, y1, x2, y2 = desenha(img, o["cls"], cx, cy)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 - x1 > 4 and y2 - y1 > 4:
                linhas.append(f"{o['cls']} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
                gt.append(f"{f + 1},{o['id']},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},1,-1,-1,-1")
        out.write(img)
        (pasta_r / nome / f"{nome}_f{f:06d}.txt").write_text("\n".join(linhas) + ("\n" if linhas else ""))
    out.release()
    (mot_dir / nome / "gt").mkdir(parents=True, exist_ok=True)
    (mot_dir / nome / "gt" / "gt.txt").write_text("\n".join(gt) + "\n")
    # gabarito da contagem: objetos cujo centro cruza o meio da imagem durante o vídeo
    cruzam = 0
    for o in objs:
        y_ini, y_fim = o["y0"], o["y0"] + (n - 1 - o["t0"]) * o["v"]
        if y_ini < h / 2 < y_fim:
            cruzam += 1
    return cruzam


def base_coco_falsa(pasta, rng, w=320, h=240, n=40):
    imgs, anns, k = [], [], 1
    (pasta / "imgs").mkdir(parents=True, exist_ok=True)
    nomes = {1: "Bottle", 2: "Plastic bag & wrapper", 3: "Styrofoam piece", 4: "Food waste"}
    for i in range(n):
        img = fundo(w, h, rng)
        for _ in range(rng.integers(1, 4)):
            c = int(rng.integers(1, 5))
            cls = {1: 0, 2: 1, 3: 2, 4: 1}[c]
            x1, y1, x2, y2 = desenha(img, cls, rng.uniform(40, w - 40), rng.uniform(30, h - 30))
            anns.append({"id": k, "image_id": i, "category_id": c, "bbox": [x1, y1, x2 - x1, y2 - y1]})
            k += 1
        nome = f"lote_1/{i:04d}.jpg"
        (pasta / "imgs" / "lote_1").mkdir(exist_ok=True)
        cv2.imwrite(str(pasta / "imgs" / nome), img)
        imgs.append({"id": i, "file_name": nome, "width": w, "height": h})
    cats = [{"id": c, "name": f"{n_} fino", "supercategory": n_} for c, n_ in nomes.items()]
    (pasta / "annotations.json").write_text(json.dumps({"images": imgs, "annotations": anns, "categories": cats}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="demo")
    ap.add_argument("--videos", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    random.seed(a.seed)
    s = Path(a.saida)
    pv, pr, pm = s / "dados/videos_brutos", s / "dados/rotulos", s / "anotacoes_mot"
    for p in (pv, pr, pm):
        p.mkdir(parents=True, exist_ok=True)
    manual = ["video,observador1,observador2"]
    for i in range(a.videos):
        nome = f"sintetico_{i + 1:02d}"
        c = gerar_video(nome, pv, pr, pm, 320, 240, 30, 8, rng)
        manual.append(f"{nome},{c},{c}")
        print(f"{nome}: {c} objetos cruzam o meio da imagem")
    (s / "dados/contagem_manual.csv").write_text("\n".join(manual) + "\n")
    base_coco_falsa(s / "bases/falsa_coco", rng)
    print(f"Dados de teste em {s.resolve()}")


if __name__ == "__main__":
    main()
