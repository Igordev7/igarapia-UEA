"""Converte o TACO (anotações oficiais no formato COCO) para o formato YOLO com as classes do IgarapIA.

Uso (após baixar o TACO com o download.py do repositório oficial):
    python scripts/preparar_taco.py --anotacoes TACO/data/annotations.json --imagens TACO/data --saida dados/igarapia

A pasta de imagens pode estar no formato oficial (batch_1/000006.jpg) ou "achatada" (1_000006.jpg).
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageOps

RAIZ = Path(__file__).resolve().parents[1]


def localizar_imagem(pasta: Path, nome_oficial: str) -> Path | None:
    candidatos = [pasta / nome_oficial, pasta / nome_oficial.replace("batch_", "").replace("/", "_")]
    achatado = nome_oficial.replace("batch_", "").replace("/", "_")
    candidatos += list(pasta.rglob(achatado))[:1]
    for c in candidatos:
        if c.exists():
            return c
        for ext in (".jpg", ".JPG", ".jpeg", ".png"):
            alt = c.with_suffix(ext)
            if alt.exists():
                return alt
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anotacoes", required=True, help="annotations.json oficial do TACO")
    ap.add_argument("--imagens", required=True, help="pasta com as imagens do TACO")
    ap.add_argument("--saida", default=str(RAIZ / "dados" / "igarapia"))
    ap.add_argument("--config", default=str(RAIZ / "config" / "classes.yaml"))
    ap.add_argument("--tam-max", type=int, default=1280, help="maior lado das imagens após redução")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--teste", type=float, default=0.15)
    ap.add_argument("--semente", type=int, default=42)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    classes = cfg["classes"]
    nome_para_classe = {}
    for classe, nomes in cfg["mapeamento_taco"].items():
        for n in nomes:
            nome_para_classe[n.lower()] = classe
    ignorar = {n.lower() for n in cfg.get("ignorar_taco", [])}

    coco = json.load(open(a.anotacoes, encoding="utf-8"))
    cat_para_idx = {}
    for c in coco["categories"]:
        n = c["name"].lower()
        if n in ignorar:
            continue
        cat_para_idx[c["id"]] = classes.index(nome_para_classe.get(n, "outros"))
    anots = defaultdict(list)
    for an in coco["annotations"]:
        anots[an["image_id"]].append(an)

    pasta_img = Path(a.imagens)
    disponiveis, faltando = [], 0
    for im in coco["images"]:
        p = localizar_imagem(pasta_img, im["file_name"])
        if p is None:
            faltando += 1
        else:
            disponiveis.append((im, p))
    if not disponiveis:
        sys.exit("Nenhuma imagem encontrada. Verifique o caminho em --imagens.")

    random.Random(a.semente).shuffle(disponiveis)
    n = len(disponiveis)
    n_te, n_va = int(n * a.teste), int(n * a.val)
    divisoes = {"test": disponiveis[:n_te], "val": disponiveis[n_te:n_te + n_va], "train": disponiveis[n_te + n_va:]}

    saida = Path(a.saida)
    estat = {s: Counter() for s in divisoes}
    for split, itens in divisoes.items():
        (saida / "images" / split).mkdir(parents=True, exist_ok=True)
        (saida / "labels" / split).mkdir(parents=True, exist_ok=True)
        for k, (im, caminho) in enumerate(itens, 1):
            img = ImageOps.exif_transpose(Image.open(caminho)).convert("RGB")
            W, H = img.size
            if (W, H) != (im["width"], im["height"]):
                print(f"Aviso: tamanho diferente do esperado em {caminho} ({W}x{H}); imagem ignorada.")
                continue
            escala = min(1.0, a.tam_max / max(W, H))
            if escala < 1:
                img = img.resize((round(W * escala), round(H * escala)), Image.LANCZOS)
            nome = im["file_name"].replace("batch_", "").replace("/", "_").rsplit(".", 1)[0]
            img.save(saida / "images" / split / f"{nome}.jpg", quality=92)
            linhas = []
            for an in anots[im["id"]]:
                if an["category_id"] not in cat_para_idx:
                    continue
                x, y, w, h = an["bbox"]
                if w <= 1 or h <= 1:
                    continue
                idx = cat_para_idx[an["category_id"]]
                linhas.append(f"{idx} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
                estat[split][classes[idx]] += 1
            (saida / "labels" / split / f"{nome}.txt").write_text("\n".join(linhas))
            if k % 100 == 0:
                print(f"  {split}: {k}/{len(itens)} imagens")

    dados = {"path": str(saida.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
             "names": {i: c for i, c in enumerate(classes)}}
    yaml.safe_dump(dados, open(saida / "data.yaml", "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
    resumo = {"imagens_encontradas": n, "imagens_nao_encontradas": faltando,
              "imagens_por_divisao": {s: len(v) for s, v in divisoes.items()},
              "objetos_por_classe": {s: dict(c) for s, c in estat.items()}}
    json.dump(resumo, open(saida / "estatisticas.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"\nImagens usadas: {n} (não encontradas: {faltando})")
    print(f"{'classe':<24}" + "".join(f"{s:>8}" for s in divisoes))
    for c in classes:
        print(f"{c:<24}" + "".join(f"{estat[s][c]:>8}" for s in divisoes))
    print(f"\nArquivo de configuração: {saida / 'data.yaml'}")


if __name__ == "__main__":
    main()
