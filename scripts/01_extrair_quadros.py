"""Extrai quadros dos vídeos gravados nos igarapés para anotação.

Exemplo:
    python scripts/01_extrair_quadros.py --videos dados/videos_brutos --saida dados/quadros --passo 15

Cada vídeo gera uma pasta com o nome do arquivo (ex.: mindu_2026-11-10_0830),
e cada quadro recebe o nome <video>_f<numero>.jpg. Com vídeos a 30 quadros por
segundo, --passo 15 salva dois quadros por segundo.
"""
import argparse
from pathlib import Path

import cv2

EXTENSOES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def extrair(video: Path, saida: Path, passo: int, max_lado: int | None) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"  [aviso] não foi possível abrir {video.name}")
        return 0
    pasta = saida / video.stem
    pasta.mkdir(parents=True, exist_ok=True)
    i = salvos = 0
    while True:
        ok, quadro = cap.read()
        if not ok:
            break
        if i % passo == 0:
            if max_lado:
                h, w = quadro.shape[:2]
                esc = max_lado / max(h, w)
                if esc < 1:
                    quadro = cv2.resize(quadro, (round(w * esc), round(h * esc)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(pasta / f"{video.stem}_f{i:06d}.jpg"), quadro, [cv2.IMWRITE_JPEG_QUALITY, 95])
            salvos += 1
        i += 1
    cap.release()
    return salvos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True, help="pasta com os vídeos (ou um único arquivo)")
    ap.add_argument("--saida", default="dados/quadros")
    ap.add_argument("--passo", type=int, default=15, help="salva 1 a cada N quadros (padrão: 15)")
    ap.add_argument("--max-lado", type=int, default=None, help="reduz os quadros para este lado máximo em pixels (opcional)")
    a = ap.parse_args()

    origem = Path(a.videos)
    videos = [origem] if origem.is_file() else sorted(p for p in origem.iterdir() if p.suffix.lower() in EXTENSOES)
    if not videos:
        raise SystemExit(f"Nenhum vídeo encontrado em {origem}")
    total = 0
    for v in videos:
        n = extrair(v, Path(a.saida), a.passo, a.max_lado)
        print(f"{v.name}: {n} quadros")
        total += n
    print(f"Total: {total} quadros em {a.saida}")


if __name__ == "__main__":
    main()
