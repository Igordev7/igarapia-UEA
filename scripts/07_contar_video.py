"""Detecta, rastreia (ByteTrack) e conta os resíduos que cruzam uma linha virtual no vídeo.

Teste rápido, antes de treinar, com o modelo genérico do COCO (só conta garrafas e copos):
    python scripts/07_contar_video.py --video meu_video.mp4 --modelo yolo11n.pt --classes-coco bottle cup

Com o modelo treinado do projeto:
    python scripts/07_contar_video.py --video dados/videos_brutos/mindu_01.mp4 \
        --modelo runs/E2_yolo11n/weights/best.pt --linha 0 0.6 1 0.6 --inicio "2026-11-10 08:30:00"

A linha é dada por dois pontos em coordenadas normalizadas (0 a 1): x1 y1 x2 y2.
O padrão (0 0.5 1 0.5) é uma linha horizontal no meio da imagem, adequada quando
a câmera está numa ponte e a correnteza "desce" ou "sobe" na imagem.

Saídas em resultados/<nome_do_video>/:
    video_anotado.mp4   caixas, identificadores, linha e contador
    contagens.csv       uma linha por objeto contado (id, classe, quadro, tempo)
    resumo.csv          total por classe e por intervalo de tempo
    rastreamento_mot.txt  (com --salvar-mot) trajetórias no formato MOT, para o 09_avaliar_rastreamento.py
"""
import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import cv2

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from igarapia.contagem import ContadorLinha, LinhaVirtual  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def desenhar(quadro, linha: LinhaVirtual, totais: Counter, nomes: dict):
    p1, p2 = (int(linha.x1), int(linha.y1)), (int(linha.x2), int(linha.y2))
    cv2.line(quadro, p1, p2, (0, 215, 255), 3)
    h = quadro.shape[0]
    esc = max(0.6, h / 1080)
    texto = [f"Total: {sum(totais.values())}"] + [f"{nomes[c]}: {n}" for c, n in sorted(totais.items())]
    y = int(40 * esc)
    for t in texto:
        cv2.putText(quadro, t, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0 * esc, (0, 0, 0), int(6 * esc), cv2.LINE_AA)
        cv2.putText(quadro, t, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0 * esc, (255, 255, 255), int(2 * esc), cv2.LINE_AA)
        y += int(38 * esc)
    return quadro


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--modelo", default="yolo11n.pt")
    ap.add_argument("--linha", type=float, nargs=4, default=[0, 0.5, 1, 0.5], metavar=("X1", "Y1", "X2", "Y2"))
    ap.add_argument("--sentido", choices=["ambos", "positivo", "negativo"], default="ambos",
                    help="conta só um sentido de travessia (veja o README para escolher)")
    ap.add_argument("--conf", type=float, default=0.1,
                    help="confiança mínima enviada ao rastreador (0,1 permite ao ByteTrack usar as caixas fracas)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--rastreador", default=str(RAIZ / "config/bytetrack_igarapia.yaml"))
    ap.add_argument("--classes-coco", nargs="*", default=None,
                    help="para o teste com o modelo do COCO: nomes das classes a manter (ex.: bottle cup)")
    ap.add_argument("--vid-stride", type=int, default=1, help="processa 1 a cada N quadros (acelera, mas piora o rastreamento)")
    ap.add_argument("--inicio", default=None, help='horário do início da gravação, "AAAA-MM-DD HH:MM:SS"')
    ap.add_argument("--intervalo-min", type=float, default=1.0, help="tamanho do intervalo do resumo, em minutos")
    ap.add_argument("--saida", default="resultados")
    ap.add_argument("--salvar-mot", action="store_true")
    ap.add_argument("--sem-video", action="store_true", help="não grava o vídeo anotado (mais rápido)")
    ap.add_argument("--dispositivo", default=None)
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"Não foi possível abrir {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    largura, altura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_quadros = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    modelo = YOLO(a.modelo)
    nomes = modelo.names
    classes = None
    if a.classes_coco:
        inv = {v: k for k, v in nomes.items()}
        faltam = [c for c in a.classes_coco if c not in inv]
        if faltam:
            raise SystemExit(f"Classes inexistentes no modelo: {faltam}. Disponíveis: {list(inv)}")
        classes = [inv[c] for c in a.classes_coco]

    linha = LinhaVirtual.de_normalizada(a.linha, largura, altura)
    contador = ContadorLinha(linha, sentido=a.sentido)
    inicio = datetime.strptime(a.inicio, "%Y-%m-%d %H:%M:%S") if a.inicio else None

    pasta = Path(a.saida) / Path(a.video).stem
    pasta.mkdir(parents=True, exist_ok=True)
    escritor = None
    if not a.sem_video:
        fps_saida = fps / a.vid_stride
        escritor = cv2.VideoWriter(str(pasta / "video_anotado.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps_saida, (largura, altura))
    mot = open(pasta / "rastreamento_mot.txt", "w") if a.salvar_mot else None

    print(f"{a.video}: {largura}x{altura}, {fps:.1f} qps, {n_quadros} quadros")
    resultados = modelo.track(source=a.video, stream=True, persist=True, tracker=a.rastreador, conf=a.conf,
                              imgsz=a.imgsz, classes=classes, vid_stride=a.vid_stride, device=a.dispositivo, verbose=False)
    for k, r in enumerate(resultados):
        quadro_idx = k * a.vid_stride
        b = r.boxes
        if b is not None and b.id is not None and len(b):
            ids = b.id.int().tolist()
            xyxy = b.xyxy.tolist()
            cls = b.cls.int().tolist()
            centros = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in xyxy]
            contador.atualizar(ids, centros, cls, quadro_idx)
            if mot:
                for i, (x1, y1, x2, y2), c in zip(ids, xyxy, b.conf.tolist()):
                    mot.write(f"{quadro_idx + 1},{i},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},{c:.3f},-1,-1,-1\n")
        if escritor:
            escritor.write(desenhar(r.plot(line_width=2), linha, Counter(contador.total_por_classe()), nomes))
        if k % 300 == 0:
            print(f"  quadro {quadro_idx}/{n_quadros}  contados: {len(contador.contagens)}")

    if escritor:
        escritor.release()
    if mot:
        mot.close()

    # contagens individuais
    with open(pasta / "contagens.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "classe", "quadro", "tempo_s", "horario", "sentido"])
        for c in contador.contagens:
            t = c.quadro / fps
            hora = (inicio + timedelta(seconds=t)).strftime("%Y-%m-%d %H:%M:%S") if inicio else ""
            w.writerow([c.id, nomes[c.classe], c.quadro, f"{t:.2f}", hora, c.sentido])

    # resumo por classe e por intervalo
    por_intervalo = defaultdict(Counter)
    for c in contador.contagens:
        por_intervalo[int(c.quadro / fps // (a.intervalo_min * 60))][nomes[c.classe]] += 1
    nomes_cls = sorted({nomes[c.classe] for c in contador.contagens})
    with open(pasta / "resumo.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["inicio_intervalo", "fim_intervalo", *nomes_cls, "total"])
        for i in sorted(por_intervalo):
            s0, s1 = i * a.intervalo_min * 60, (i + 1) * a.intervalo_min * 60
            fmt = (lambda s: (inicio + timedelta(seconds=s)).strftime("%H:%M:%S")) if inicio else (lambda s: f"{s:.0f}s")
            cont = por_intervalo[i]
            w.writerow([fmt(s0), fmt(s1), *[cont[n] for n in nomes_cls], sum(cont.values())])
        tot = Counter(nomes[c.classe] for c in contador.contagens)
        w.writerow(["TOTAL", "", *[tot[n] for n in nomes_cls], sum(tot.values())])

    print(f"\nObjetos contados: {len(contador.contagens)}")
    for n, q in Counter(nomes[c.classe] for c in contador.contagens).most_common():
        print(f"  {n}: {q}")
    print(f"Resultados em {pasta.resolve()}{os.sep}")


if __name__ == "__main__":
    main()
