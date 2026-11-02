"""Desfoca pessoas e veículos (e regiões fixas, se quiser) de um vídeo antes de
divulgar imagens, conforme a LGPD.

Usa o modelo genérico do COCO (yolo11n.pt) para achar pessoas, carros, motos,
ônibus e caminhões, e desfoca a caixa inteira de cada um.

Exemplos:
    python scripts/anonimizar_video.py --video bruto.mp4 --saida anonimo.mp4
    # também desfoca uma faixa fixa da imagem (x1 y1 x2 y2 de 0 a 1), ex.: a rua no topo
    python scripts/anonimizar_video.py --video bruto.mp4 --saida anonimo.mp4 --regiao 0 0 1 0.2

Confira o vídeo resultante antes de publicar: nenhum detector acerta 100%.
"""
import argparse

import cv2
from ultralytics import YOLO

PESSOAS_VEICULOS = [0, 1, 2, 3, 5, 7]  # COCO: person, bicycle, car, motorcycle, bus, truck


def borrar(q, x0, y0, x1, y1, k):
    if x1 > x0 and y1 > y0:
        q[y0:y1, x0:x1] = cv2.GaussianBlur(q[y0:y1, x0:x1], (k, k), 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--modelo", default="yolo11n.pt", help="modelo do COCO usado para achar pessoas e veículos")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--regiao", type=float, nargs=4, action="append", default=[], metavar=("X1", "Y1", "X2", "Y2"))
    ap.add_argument("--dispositivo", default=None)
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"Não foi possível abrir {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    out = cv2.VideoWriter(a.saida, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    k = max(31, (min(w, h) // 15) | 1)  # tamanho ímpar do desfoque

    modelo = YOLO(a.modelo)
    n = achados = 0
    for r in modelo.predict(source=a.video, stream=True, classes=PESSOAS_VEICULOS, conf=a.conf, device=a.dispositivo, verbose=False):
        q = r.orig_img.copy()
        for x0, y0, x1, y1 in r.boxes.xyxy.int().tolist():
            m = int(0.1 * (x1 - x0))
            borrar(q, max(0, x0 - m), max(0, y0 - m), min(w, x1 + m), min(h, y1 + m), k)
            achados += 1
        for (rx1, ry1, rx2, ry2) in a.regiao:
            borrar(q, int(rx1 * w), int(ry1 * h), int(rx2 * w), int(ry2 * h), k)
        out.write(q)
        n += 1
    out.release()
    print(f"{n} quadros processados, {achados} áreas desfocadas -> {a.saida}")


if __name__ == "__main__":
    main()
