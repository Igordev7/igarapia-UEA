"""Converte bases públicas (formato COCO ou Pascal VOC) para o formato YOLO,
já com as classes do IgarapIA e dividida em treino e validação.

Exemplos:
    # TACO e UAVVaste vêm em COCO (um arquivo .json)
    python scripts/02_converter_anotacoes.py --formato coco --fonte taco \
        --anotacoes bases/TACO/data/annotations.json --imagens bases/TACO/data \
        --saida dados/publicos/taco

    # Bases em Pascal VOC (uma pasta com um .xml por imagem)
    python scripts/02_converter_anotacoes.py --formato voc --fonte flow \
        --anotacoes bases/FloW_IMG/training/annotations --imagens bases/FloW_IMG/training/images \
        --saida dados/publicos/flow

Antes de converter de verdade, rode com --so-listar para ver os nomes das classes
da base e ajustar config/mapeamento_classes.yaml.
"""
import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[1]


def ler_coco(arq_json: Path, usar_super: bool):
    dados = json.loads(arq_json.read_text(encoding="utf-8"))
    cats = {c["id"]: (c.get("supercategory") if usar_super and c.get("supercategory") else c["name"]) for c in dados["categories"]}
    imgs = {im["id"]: {"arquivo": im["file_name"], "w": im["width"], "h": im["height"], "objs": []} for im in dados["images"]}
    for an in dados["annotations"]:
        if an["image_id"] in imgs and an.get("bbox"):
            x, y, w, h = an["bbox"]
            imgs[an["image_id"]]["objs"].append((cats[an["category_id"]], x, y, x + w, y + h))
    return list(imgs.values())


def ler_voc(pasta: Path):
    itens = []
    for xml in sorted(pasta.glob("*.xml")):
        r = ET.parse(xml).getroot()
        tam = r.find("size")
        w, h = float(tam.findtext("width")), float(tam.findtext("height"))
        nome = r.findtext("filename") or (xml.stem + ".jpg")
        objs = []
        for o in r.findall("object"):
            b = o.find("bndbox")
            objs.append((o.findtext("name").strip(), *(float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))))
        itens.append({"arquivo": nome, "w": w, "h": h, "objs": objs})
    return itens


def para_yolo(obj, w, h, idx):
    _, x1, y1, x2, y2 = obj
    x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
    y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return f"{idx} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--formato", choices=["coco", "voc"], required=True)
    ap.add_argument("--fonte", required=True, help="nome da base em config/mapeamento_classes.yaml (taco, uavvaste, flow...)")
    ap.add_argument("--anotacoes", required=True, help="arquivo .json (COCO) ou pasta de .xml (VOC)")
    ap.add_argument("--imagens", required=True, help="pasta-base das imagens")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--mapeamento", default=str(RAIZ / "config/mapeamento_classes.yaml"))
    ap.add_argument("--val", type=float, default=0.1, help="fração para validação (padrão: 0,1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--descartar-vazias", action="store_true", help="não copia imagens sem nenhum objeto mapeado")
    ap.add_argument("--so-listar", action="store_true", help="apenas mostra as classes da base e sai")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.mapeamento).read_text(encoding="utf-8"))
    classes = cfg["classes"]
    fonte = cfg["fontes"].get(a.fonte, {"padrao": "outros_residuos", "mapa": {}})
    mapa, padrao = fonte.get("mapa") or {}, fonte.get("padrao")

    itens = ler_coco(Path(a.anotacoes), fonte.get("usar_supercategoria", False)) if a.formato == "coco" else ler_voc(Path(a.anotacoes))
    origem = Counter(o[0] for it in itens for o in it["objs"])
    print(f"{len(itens)} imagens, {sum(origem.values())} objetos. Classes na base:")
    for nome, n in origem.most_common():
        destino = mapa[nome] if nome in mapa else padrao
        print(f"  {nome!r:35} {n:6}  ->  {destino}")
    if a.so_listar:
        return

    random.Random(a.seed).shuffle(itens)
    n_val = max(1, round(len(itens) * a.val))
    saida = Path(a.saida)
    contagem, faltando, copiadas = Counter(), 0, 0
    for i, it in enumerate(itens):
        split = "val" if i < n_val else "train"
        linhas = []
        for o in it["objs"]:
            destino = mapa[o[0]] if o[0] in mapa else padrao
            if destino is None:
                continue
            ln = para_yolo(o, it["w"], it["h"], classes.index(destino))
            if ln:
                linhas.append(ln)
                contagem[destino] += 1
        if a.descartar_vazias and not linhas:
            continue
        src = Path(a.imagens) / it["arquivo"]
        if not src.exists():
            faltando += 1
            continue
        nome = it["arquivo"].replace("/", "_").replace("\\", "_")
        (saida / split / "images").mkdir(parents=True, exist_ok=True)
        (saida / split / "labels").mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, saida / split / "images" / nome)
        (saida / split / "labels" / (Path(nome).stem + ".txt")).write_text("\n".join(linhas) + ("\n" if linhas else ""))
        copiadas += 1

    print(f"\nImagens copiadas: {copiadas} (validação: {n_val}); não encontradas: {faltando}")
    print("Caixas por classe do IgarapIA:", dict(contagem))


if __name__ == "__main__":
    main()
