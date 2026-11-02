"""Gera os arquivos de dados do YOLO para os três cenários do projeto.

    E1: treino só com bases públicas (validação também pública)
    E2: treino com bases públicas + vídeos locais (validação local)
    E3: treino só com vídeos locais (validação local)

Os três usam o MESMO conjunto de teste: dados/local/test.

Exemplo:
    python scripts/04_montar_cenarios.py --publicos dados/publicos --local dados/local --saida cenarios
"""
import argparse
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--publicos", default="dados/publicos", help="pasta com uma subpasta por base convertida")
    ap.add_argument("--local", default="dados/local")
    ap.add_argument("--saida", default="cenarios")
    ap.add_argument("--mapeamento", default=str(RAIZ / "config/mapeamento_classes.yaml"))
    a = ap.parse_args()

    classes = yaml.safe_load(Path(a.mapeamento).read_text(encoding="utf-8"))["classes"]
    nomes = dict(enumerate(classes))
    pub = Path(a.publicos).resolve()
    loc = Path(a.local).resolve()
    bases = sorted(p for p in pub.iterdir() if (p / "train" / "images").is_dir()) if pub.is_dir() else []
    if not (loc / "test" / "images").is_dir():
        raise SystemExit(f"Não encontrei {loc}/test/images. Rode antes o 03_dividir_por_video.py.")

    pub_tr = [str(b / "train" / "images") for b in bases]
    pub_va = [str(b / "val" / "images") for b in bases if (b / "val" / "images").is_dir()]
    loc_tr, loc_va, loc_te = str(loc / "train" / "images"), str(loc / "val" / "images"), str(loc / "test" / "images")

    cenarios = {}
    if bases:
        cenarios["E1"] = {"train": pub_tr, "val": pub_va or pub_tr}
        cenarios["E2"] = {"train": pub_tr + [loc_tr], "val": [loc_va]}
    else:
        print("[aviso] nenhuma base pública encontrada: gerando apenas o E3")
    cenarios["E3"] = {"train": [loc_tr], "val": [loc_va]}

    saida = Path(a.saida)
    saida.mkdir(parents=True, exist_ok=True)
    for nome, c in cenarios.items():
        cfg = {"train": c["train"], "val": c["val"], "test": [loc_te], "names": nomes}
        arq = saida / f"{nome}.yaml"
        arq.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"{arq}: treino = {len(c['train'])} pasta(s), validação = {len(c['val'])} pasta(s)")


if __name__ == "__main__":
    main()
