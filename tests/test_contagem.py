"""Testes da regra de contagem. Rode com:  python tests/test_contagem.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from igarapia.contagem import ContadorLinha, LinhaVirtual  # noqa: E402


def linha_horizontal():
    # linha em y = 50, de x = 0 a x = 100
    return LinhaVirtual(0, 50, 100, 50)


def test_conta_uma_vez_ao_cruzar():
    c = ContadorLinha(linha_horizontal())
    for q, y in enumerate([10, 30, 45, 55, 70, 90]):
        c.atualizar([1], [(50, y)], [0], q)
    assert len(c.contagens) == 1
    assert c.contagens[0].quadro == 3


def test_vai_e_volta_nao_conta_duas_vezes():
    c = ContadorLinha(linha_horizontal())
    for q, y in enumerate([40, 60, 40, 60]):
        c.atualizar([7], [(50, y)], [1], q)
    assert len(c.contagens) == 1


def test_sentido():
    # descendo na imagem (y crescente) com a linha da esquerda para a direita => lado positivo
    desce = ContadorLinha(linha_horizontal(), sentido="positivo")
    sobe = ContadorLinha(linha_horizontal(), sentido="positivo")
    for q, y in enumerate([40, 60]):
        desce.atualizar([1], [(50, y)], [0], q)
    for q, y in enumerate([60, 40]):
        sobe.atualizar([1], [(50, y)], [0], q)
    assert len(desce.contagens) == 1
    assert len(sobe.contagens) == 0


def test_fora_do_segmento_nao_conta():
    c = ContadorLinha(linha_horizontal())
    for q, y in enumerate([40, 60]):
        c.atualizar([1], [(300, y)], [0], q)
    assert len(c.contagens) == 0


def test_varios_objetos_e_classes():
    c = ContadorLinha(linha_horizontal())
    c.atualizar([1, 2, 3], [(10, 40), (20, 40), (30, 40)], [0, 2, 2], 0)
    c.atualizar([1, 2, 3], [(10, 60), (20, 60), (30, 45)], [0, 2, 2], 1)
    assert c.total_por_classe() == {0: 1, 2: 1}


def test_objeto_sobre_a_linha():
    c = ContadorLinha(linha_horizontal())
    for q, y in enumerate([40, 50, 60]):
        c.atualizar([1], [(50, y)], [0], q)
    assert len(c.contagens) == 1


if __name__ == "__main__":
    testes = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in testes:
        t()
        print("ok", t.__name__)
    print(f"{len(testes)} testes passaram")
