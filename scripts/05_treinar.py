"""Treina (ajusta) o detector YOLO11 em um dos cenários.

Exemplos:
    python scripts/05_treinar.py --dados cenarios/E2.yaml --modelo yolo11n.pt --nome E2_yolo11n
    python scripts/05_treinar.py --dados cenarios/E2.yaml --modelo yolo11s.pt --nome E2_yolo11s --epocas 100

O modelo parte dos pesos pré-treinados no COCO (baixados automaticamente na
primeira execução). O melhor modelo fica em runs/<nome>/weights/best.pt.
No Google Colab, ative a GPU em Ambiente de execução > Alterar o tipo de ambiente.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dados", required=True, help="arquivo do cenário (ex.: cenarios/E2.yaml)")
    ap.add_argument("--modelo", default="yolo11n.pt", help="yolo11n.pt (leve) ou yolo11s.pt")
    ap.add_argument("--nome", required=True, help="nome do experimento (ex.: E2_yolo11n)")
    ap.add_argument("--epocas", type=int, default=100)
    ap.add_argument("--paciencia", type=int, default=20, help="para se a validação não melhorar por N épocas")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lote", type=int, default=16, help="tamanho do lote; reduza se faltar memória da GPU")
    ap.add_argument("--dispositivo", default=None, help="0 para a primeira GPU, cpu para CPU (padrão: automático)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    modelo = YOLO(a.modelo)
    modelo.train(
        data=a.dados,
        epochs=a.epocas,
        patience=a.paciencia,
        imgsz=a.imgsz,
        batch=a.lote,
        device=a.dispositivo,
        seed=a.seed,
        project=str(Path("runs").resolve()),
        name=a.nome,
        exist_ok=True,
        # aumento de dados citado no projeto (valores próximos aos padrões da biblioteca)
        hsv_v=0.4,   # variação de brilho
        fliplr=0.5,  # espelhamento horizontal
        flipud=0.0,  # a câmera é fixa: não inverter de cabeça para baixo
        mosaic=1.0,
        plots=True,
    )
    melhor = Path("runs") / a.nome / "weights" / "best.pt"
    print(f"\nMelhor modelo: {melhor.resolve()}")


if __name__ == "__main__":
    main()
