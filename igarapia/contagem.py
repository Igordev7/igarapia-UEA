"""Contagem de objetos rastreados que cruzam uma linha virtual.

Esta parte não depende do YOLO: recebe, a cada quadro, os identificadores,
os centros das caixas e as classes vindos do rastreador e decide quando cada
objeto deve ser contado. Assim ela pode ser testada isoladamente
(veja tests/test_contagem.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LinhaVirtual:
    """Segmento de reta entre (x1, y1) e (x2, y2), em pixels."""

    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def de_normalizada(cls, coords, largura: int, altura: int) -> "LinhaVirtual":
        """Cria a linha a partir de coordenadas entre 0 e 1."""
        x1, y1, x2, y2 = coords
        return cls(x1 * largura, y1 * altura, x2 * largura, y2 * altura)

    def lado(self, x: float, y: float) -> float:
        """Produto vetorial: > 0 de um lado da linha, < 0 do outro, 0 sobre ela."""
        return (self.x2 - self.x1) * (y - self.y1) - (self.y2 - self.y1) * (x - self.x1)

    def projecao(self, x: float, y: float) -> float:
        """Posição da projeção do ponto sobre o segmento (0 = início, 1 = fim)."""
        dx, dy = self.x2 - self.x1, self.y2 - self.y1
        n2 = dx * dx + dy * dy
        if n2 == 0:
            return 0.0
        return ((x - self.x1) * dx + (y - self.y1) * dy) / n2


@dataclass
class Contagem:
    id: int
    classe: int
    quadro: int
    sentido: str  # "positivo" ou "negativo"


@dataclass
class ContadorLinha:
    """Conta cada identificador uma única vez quando ele cruza a linha.

    sentido: "ambos", "positivo" (do lado negativo para o positivo)
             ou "negativo" (do lado positivo para o negativo).
    margem:  tolerância, em fração do comprimento da linha, para aceitar
             cruzamentos um pouco além das pontas do segmento.
    """

    linha: LinhaVirtual
    sentido: str = "ambos"
    margem: float = 0.05
    _ultimo_lado: dict = field(default_factory=dict)
    _contados: set = field(default_factory=set)
    contagens: list = field(default_factory=list)

    def __post_init__(self):
        if self.sentido not in ("ambos", "positivo", "negativo"):
            raise ValueError("sentido deve ser 'ambos', 'positivo' ou 'negativo'")

    def atualizar(self, ids, centros, classes, quadro: int) -> list[Contagem]:
        """Processa as detecções de um quadro e devolve as novas contagens."""
        novas = []
        for id_, (cx, cy), cls in zip(ids, centros, classes):
            id_ = int(id_)
            lado = self.linha.lado(cx, cy)
            anterior = self._ultimo_lado.get(id_)
            if lado != 0:
                self._ultimo_lado[id_] = lado
            if anterior is None or lado == 0 or id_ in self._contados:
                continue
            if (anterior < 0) == (lado < 0):
                continue  # não trocou de lado
            t = self.linha.projecao(cx, cy)
            if t < -self.margem or t > 1 + self.margem:
                continue  # passou fora do segmento
            sentido = "positivo" if lado > 0 else "negativo"
            if self.sentido != "ambos" and sentido != self.sentido:
                continue
            self._contados.add(id_)
            c = Contagem(id=id_, classe=int(cls), quadro=quadro, sentido=sentido)
            self.contagens.append(c)
            novas.append(c)
        return novas

    def total_por_classe(self) -> dict[int, int]:
        tot: dict[int, int] = {}
        for c in self.contagens:
            tot[c.classe] = tot.get(c.classe, 0) + 1
        return tot
