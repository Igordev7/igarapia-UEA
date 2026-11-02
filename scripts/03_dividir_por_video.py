"""Divide os quadros anotados dos igarapés em treino, validação e teste POR VÍDEO.

A divisão por vídeo evita que quadros quase iguais do mesmo vídeo fiquem no
treino e no teste ao mesmo tempo, o que inflaria os resultados.

Estrutura esperada:
    dados/quadros/<video>/<video>_f000000.jpg     (saída do 01_extrair_quadros.py)
    dados/rotulos/<video>/<video>_f000000.txt     (rótulos YOLO exportados do CVAT ou Label Studio)

Só entram quadros que têm arquivo de rótulo (um .txt vazio significa "quadro sem lixo").

Exemplo:
    python scripts/03_dividir_por_video.py --quadros dados/quadros --rotulos dados/rotulos --saida dados/local
"""
import argparse
import json
import random
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quadros", default="dados/quadros")
    ap.add_argument("--rotulos", default="dados/rotulos")
    ap.add_argument("--saida", default="dados/local")
    ap.add_argument("--proporcao", type=float, nargs=3, default=[0.70, 0.15, 0.15], metavar=("TREINO", "VAL", "TESTE"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teste", nargs="*", default=None, help="(opcional) nomes dos vídeos que devem ir para o teste")
    a = ap.parse_args()

    q, r, s = Path(a.quadros), Path(a.rotulos), Path(a.saida)
    videos = sorted(p.name for p in q.iterdir() if p.is_dir() and (r / p.name).is_dir())
    if len(videos) < 3:
        raise SystemExit(f"São necessários pelo menos 3 vídeos anotados; encontrei {len(videos)}: {videos}")

    if a.teste:
        teste = [v for v in videos if v in a.teste]
        resto = [v for v in videos if v not in teste]
        random.Random(a.seed).shuffle(resto)
        n_val = max(1, round(len(videos) * a.proporcao[1]))
        grupos = {"test": teste, "val": resto[:n_val], "train": resto[n_val:]}
    else:
        random.Random(a.seed).shuffle(videos)
        n = len(videos)
        n_te = max(1, round(n * a.proporcao[2]))
        n_va = max(1, round(n * a.proporcao[1]))
        grupos = {"test": videos[:n_te], "val": videos[n_te:n_te + n_va], "train": videos[n_te + n_va:]}

    resumo = {}
    for split, vids in grupos.items():
        (s / split / "images").mkdir(parents=True, exist_ok=True)
        (s / split / "labels").mkdir(parents=True, exist_ok=True)
        n_img = n_obj = 0
        for v in vids:
            for img in sorted((q / v).glob("*.jpg")):
                rot = r / v / (img.stem + ".txt")
                if not rot.exists():
                    continue
                shutil.copy2(img, s / split / "images" / img.name)
                shutil.copy2(rot, s / split / "labels" / rot.name)
                n_img += 1
                n_obj += sum(1 for ln in rot.read_text().splitlines() if ln.strip())
        resumo[split] = {"videos": vids, "quadros": n_img, "objetos": n_obj}
        print(f"{split:5}: {len(vids)} vídeos, {n_img} quadros, {n_obj} objetos")

    (s / "divisao.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False))
    print(f"Divisão registrada em {s / 'divisao.json'}")


if __name__ == "__main__":
    main()
