"""Treina o detector YOLO com a base preparada.

Uso:
    python scripts/treinar.py --dados dados/igarapia/data.yaml --epocas 60
No Google Colab com GPU, use --dispositivo 0.
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dados", required=True, help="data.yaml gerado pelo preparar_taco.py")
    ap.add_argument("--modelo", default="yolo11n.pt", help="pesos pré-treinados de partida")
    ap.add_argument("--epocas", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--dispositivo", default=None, help="'cpu', '0' (GPU) ou vazio para automático")
    ap.add_argument("--paciencia", type=int, default=20, help="para o treino se não houver melhora")
    ap.add_argument("--nome", default="igarapia")
    a = ap.parse_args()

    modelo = YOLO(a.modelo)
    modelo.train(data=a.dados, epochs=a.epocas, imgsz=a.imgsz, batch=a.lote, device=a.dispositivo,
                 patience=a.paciencia, project=str(RAIZ / "treinos"), name=a.nome, exist_ok=True,
                 # aumentos de dados que simulam reflexo, variação de luz e orientação na água
                 hsv_v=0.5, hsv_s=0.6, fliplr=0.5, flipud=0.2, degrees=10, mosaic=1.0)
    melhor = RAIZ / "treinos" / a.nome / "weights" / "best.pt"
    destino = RAIZ / "modelos" / "igarapia_best.pt"
    destino.parent.mkdir(exist_ok=True)
    shutil.copy(melhor, destino)
    print(f"\nModelo final copiado para: {destino}")


if __name__ == "__main__":
    main()
