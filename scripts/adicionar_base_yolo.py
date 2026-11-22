"""Acrescenta à base do IgarapIA outra base já no formato YOLO (por exemplo, exportada do Roboflow ou do CVAT).

Use para incluir fotos de lixo na água ou fotos de igarapés anotadas pela equipe.

Exemplo:
    python scripts/adicionar_base_yolo.py --origem bases/lixo_rio/data.yaml --destino dados/igarapia \
        --mapa "bottle=garrafa_pet,plastic bottle=garrafa_pet,can=lata,plastic bag=sacola_filme_plastico,styrofoam=isopor" \
        --prefixo rio

Classes da origem que não estão no --mapa vão para "outros" (ou são descartadas com --descartar-resto).
As divisões da origem (train, valid/val, test) são mantidas.
"""
import argparse
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image, ImageOps

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DIV = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origem", required=True, help="data.yaml da base a acrescentar")
    ap.add_argument("--destino", required=True, help="pasta da base do IgarapIA (com data.yaml)")
    ap.add_argument("--mapa", default="", help='"classe_origem=classe_igarapia,..."')
    ap.add_argument("--prefixo", default="extra", help="prefixo dos arquivos copiados, para não sobrescrever")
    ap.add_argument("--descartar-resto", action="store_true", help="ignora classes fora do mapa em vez de usar 'outros'")
    ap.add_argument("--tam-max", type=int, default=1280)
    a = ap.parse_args()

    destino = Path(a.destino)
    dados = yaml.safe_load(open(destino / "data.yaml", encoding="utf-8"))
    nomes_dest = dados["names"]
    nomes_dest = [nomes_dest[i] for i in sorted(nomes_dest)] if isinstance(nomes_dest, dict) else list(nomes_dest)

    ori_yaml = Path(a.origem)
    ori = yaml.safe_load(open(ori_yaml, encoding="utf-8"))
    nomes_ori = ori["names"]
    nomes_ori = [nomes_ori[i] for i in sorted(nomes_ori)] if isinstance(nomes_ori, dict) else list(nomes_ori)
    raiz_ori = Path(ori.get("path") or ori_yaml.parent)
    if not raiz_ori.is_absolute():
        raiz_ori = (ori_yaml.parent / raiz_ori).resolve()

    mapa = {}
    for par in filter(None, (p.strip() for p in a.mapa.split(","))):
        o, d = (x.strip() for x in par.split("="))
        if d not in nomes_dest:
            raise SystemExit(f"Classe de destino desconhecida: {d}. Use uma de {nomes_dest}")
        mapa[o.lower()] = nomes_dest.index(d)
    conv = {}
    for i, n in enumerate(nomes_ori):
        if n.lower() in mapa:
            conv[i] = mapa[n.lower()]
        elif not a.descartar_resto:
            conv[i] = nomes_dest.index("outros")

    total_img, cont = Counter(), Counter()
    for chave in ("train", "val", "valid", "test"):
        if chave not in ori:
            continue
        pasta_img = (raiz_ori / ori[chave]).resolve()
        if not pasta_img.exists():
            pasta_img = (ori_yaml.parent / ori[chave].replace("../", "")).resolve()
        if not pasta_img.exists():
            print(f"Aviso: pasta não encontrada para '{chave}': {pasta_img}")
            continue
        div = DIV[chave]
        (destino / "images" / div).mkdir(parents=True, exist_ok=True)
        (destino / "labels" / div).mkdir(parents=True, exist_ok=True)
        for img_path in sorted(p for p in pasta_img.rglob("*") if p.suffix.lower() in EXT):
            rot = Path(str(img_path).replace(f"{Path('images')}", f"{Path('labels')}")).with_suffix(".txt")
            linhas = []
            if rot.exists():
                for l in open(rot, encoding="utf-8"):
                    p = l.split()
                    if len(p) != 5:  # ignora polígonos de segmentação
                        continue
                    c = int(p[0])
                    if c in conv:
                        linhas.append(" ".join([str(conv[c])] + p[1:]))
                        cont[nomes_dest[conv[c]]] += 1
            img = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
            W, H = img.size
            esc = min(1.0, a.tam_max / max(W, H))
            if esc < 1:
                img = img.resize((round(W * esc), round(H * esc)), Image.LANCZOS)
            nome = f"{a.prefixo}_{img_path.stem}"
            img.save(destino / "images" / div / f"{nome}.jpg", quality=92)
            (destino / "labels" / div / f"{nome}.txt").write_text("\n".join(linhas))
            total_img[div] += 1

    print("Imagens acrescentadas:", dict(total_img))
    print("Objetos acrescentados por classe:", dict(cont))
    print("Apague os arquivos labels/*.cache antes de treinar de novo.")


if __name__ == "__main__":
    main()
