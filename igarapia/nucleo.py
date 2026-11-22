"""Funções centrais de análise: detecção em imagens e contagem com rastreamento em vídeos."""
from __future__ import annotations

import csv
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2

EXT_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXT_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}

# configuração do ByteTrack ajustada para objetos flutuantes (cai na padrão se o arquivo não existir)
_cfg = Path(__file__).resolve().parents[1] / "config" / "bytetrack_igarapia.yaml"
RASTREADOR = str(_cfg) if _cfg.exists() else "bytetrack.yaml"

CAMPOS_HISTORICO = ["data_analise", "local", "fonte", "tipo", "classe", "quantidade"]


@dataclass
class Resultado:
    fonte: str
    tipo: str  # "imagem" ou "video"
    local: str
    contagem: Counter = field(default_factory=Counter)
    deteccoes: list = field(default_factory=list)  # dicts por detecção
    saida_anotada: str | None = None
    quadros_processados: int = 0

    def resumo(self) -> dict:
        return {
            "data_analise": datetime.now().isoformat(timespec="seconds"),
            "local": self.local,
            "fonte": self.fonte,
            "tipo": self.tipo,
            "quadros_processados": self.quadros_processados,
            "total_itens": int(sum(self.contagem.values())),
            "itens_por_classe": dict(self.contagem.most_common()),
            "saida_anotada": self.saida_anotada,
        }


def carregar_modelo(caminho: str):
    from ultralytics import YOLO
    return YOLO(caminho)


def analisar_imagem(modelo, caminho: str, pasta_saida: str, conf: float = 0.25, local: str = "",
                    imgsz: int = 640, agnostico: bool = True, tta: bool = False) -> Resultado:
    """Detecta resíduos em uma imagem e conta os itens por classe.

    agnostico=True faz a supressão de caixas sem olhar a classe: o mesmo objeto não recebe
    duas caixas de classes diferentes (por exemplo, "lata" e "garrafa_pet" na mesma garrafa).
    tta=True analisa também versões espelhadas e redimensionadas da imagem (mais lento, costuma achar mais objetos).
    """
    r = modelo.predict(caminho, conf=conf, imgsz=imgsz, agnostic_nms=agnostico, augment=tta, verbose=False)[0]
    nomes = r.names
    res = Resultado(fonte=os.path.basename(caminho), tipo="imagem", local=local, quadros_processados=1)
    for cls, cf, xyxy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
        classe = nomes[int(cls)]
        res.contagem[classe] += 1
        res.deteccoes.append({"quadro": 0, "id_objeto": "", "classe": classe, "confianca": round(cf, 3),
                              "x1": round(xyxy[0]), "y1": round(xyxy[1]), "x2": round(xyxy[2]), "y2": round(xyxy[3])})
    Path(pasta_saida).mkdir(parents=True, exist_ok=True)
    saida = str(Path(pasta_saida) / f"anotada_{Path(caminho).stem}.jpg")
    cv2.imwrite(saida, r.plot())
    res.saida_anotada = saida
    return res


def analisar_video(modelo, caminho: str, pasta_saida: str, conf: float = 0.25, local: str = "",
                   passo: int = 1, ao_progredir=None, imgsz: int = 640, agnostico: bool = True) -> Resultado:
    """Rastreia os resíduos ao longo do vídeo; cada objeto é contado uma única vez (pelo ID de rastreamento)."""
    cap = cv2.VideoCapture(caminho)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    Path(pasta_saida).mkdir(parents=True, exist_ok=True)
    saida = str(Path(pasta_saida) / f"anotado_{Path(caminho).stem}.mp4")
    escritor = None
    vistos: dict[int, str] = {}  # id do objeto -> classe mais recente
    votos: dict[int, Counter] = {}
    res = Resultado(fonte=os.path.basename(caminho), tipo="video", local=local)
    i = 0
    while True:
        ok, quadro = cap.read()
        if not ok:
            break
        if i % passo == 0:
            r = modelo.track(quadro, conf=conf, imgsz=imgsz, agnostic_nms=agnostico, persist=True,
                             tracker=RASTREADOR, verbose=False)[0]
            nomes = r.names
            ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(r.boxes)
            for oid, cls, cf, xyxy in zip(ids, r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                classe = nomes[int(cls)]
                if oid is not None:
                    votos.setdefault(oid, Counter())[classe] += 1
                res.deteccoes.append({"quadro": i, "id_objeto": oid if oid is not None else "", "classe": classe,
                                      "confianca": round(cf, 3), "x1": round(xyxy[0]), "y1": round(xyxy[1]),
                                      "x2": round(xyxy[2]), "y2": round(xyxy[3])})
            anotado = r.plot()
            if escritor is None:
                h, w = anotado.shape[:2]
                escritor = cv2.VideoWriter(saida, cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps / passo), (w, h))
            escritor.write(anotado)
            res.quadros_processados += 1
        i += 1
        if ao_progredir and total:
            ao_progredir(min(1.0, i / total))
    cap.release()
    if escritor is not None:
        escritor.release()
    # cada objeto rastreado recebe a classe mais votada ao longo dos quadros
    for oid, c in votos.items():
        vistos[oid] = c.most_common(1)[0][0]
    res.contagem = Counter(vistos.values())
    res.saida_anotada = saida if escritor is not None else None
    return res


def salvar_resultado(res: Resultado, pasta_saida: str, historico: str | None = None) -> dict:
    """Grava as detecções (CSV), o resumo (JSON) e acrescenta a contagem ao histórico."""
    pasta = Path(pasta_saida)
    pasta.mkdir(parents=True, exist_ok=True)
    base = Path(res.fonte).stem
    with open(pasta / f"deteccoes_{base}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["quadro", "id_objeto", "classe", "confianca", "x1", "y1", "x2", "y2"])
        w.writeheader()
        w.writerows(res.deteccoes)
    resumo = res.resumo()
    with open(pasta / f"resumo_{base}.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)
    historico = historico or str(pasta / "historico.csv")
    novo = not os.path.exists(historico)
    with open(historico, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_HISTORICO)
        if novo:
            w.writeheader()
        for classe, qtd in (res.contagem.items() or [("nenhum_item", 0)]):
            w.writerow({"data_analise": resumo["data_analise"], "local": res.local, "fonte": res.fonte,
                        "tipo": res.tipo, "classe": classe, "quantidade": qtd})
    return resumo


def listar_fontes(fonte: str) -> list[str]:
    p = Path(fonte)
    if p.is_dir():
        return sorted(str(x) for x in p.iterdir() if x.suffix.lower() in EXT_IMAGEM | EXT_VIDEO)
    return [str(p)]


def tipo_da_fonte(caminho: str) -> str:
    ext = Path(caminho).suffix.lower()
    if ext in EXT_VIDEO:
        return "video"
    if ext in EXT_IMAGEM:
        return "imagem"
    raise ValueError(f"Formato não suportado: {caminho}")
