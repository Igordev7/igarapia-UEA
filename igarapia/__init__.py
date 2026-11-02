"""IgarapIA: detecção, rastreamento e contagem de resíduos flutuantes em vídeo."""
from .contagem import ContadorLinha, Contagem, LinhaVirtual

CLASSES = ["garrafa_plastica", "sacola_filme_plastico", "isopor", "lata_metal", "outros_residuos"]

__all__ = ["CLASSES", "ContadorLinha", "Contagem", "LinhaVirtual"]
