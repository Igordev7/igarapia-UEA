# IgarapIA – código do modelo

Implementação do sistema descrito no projeto: detector YOLO11 ajustado para resíduos
flutuantes, rastreamento com ByteTrack e contagem por linha virtual.

## 1. Instalação

**Opção A – Google Colab (recomendado, tem GPU grátis):** abra
`notebooks/IgarapIA_Colab.ipynb` no Colab e siga as células.

**Opção B – no computador (Python 3.10 ou mais novo):**

```bash
cd igarapia
python -m venv .venv
source .venv/bin/activate        # no Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tests/test_contagem.py    # confere a regra de contagem
```

Sem GPU dá para rodar a contagem em vídeo e treinos curtos; o treino completo é
bem mais rápido no Colab.

## 2. Teste rápido (antes de ter dados próprios)

O modelo genérico do COCO já reconhece garrafas (`bottle`) e copos (`cup`).
Grave um vídeo curto de uma garrafa passando (pode ser numa bacia ou num corredor)
e rode:

```bash
python scripts/07_contar_video.py --video teste.mp4 --modelo yolo11n.pt --classes-coco bottle cup
```

O resultado fica em `resultados/teste/` (vídeo anotado, `contagens.csv` e `resumo.csv`).
Isso só serve para ver o sistema funcionando; o modelo do projeto vem do passo 3.

### Teste do fluxo inteiro com dados sintéticos

Para conferir que todos os scripts rodam na sua máquina, gere vídeos de brinquedo
(com rótulos, contagem manual e gabarito de rastreamento já prontos) e siga o passo 3
dentro da pasta `demo`. Os números obtidos **não** têm valor científico.

```bash
python tests/gerar_dados_sinteticos.py --saida demo
cd demo
python ../scripts/01_extrair_quadros.py --videos dados/videos_brutos --saida dados/quadros --passo 5
python ../scripts/02_converter_anotacoes.py --formato coco --fonte taco \
    --anotacoes bases/falsa_coco/annotations.json --imagens bases/falsa_coco/imgs --saida dados/publicos/falsa
python ../scripts/03_dividir_por_video.py
python ../scripts/04_montar_cenarios.py --mapeamento ../config/mapeamento_classes.yaml
python ../scripts/05_treinar.py --dados cenarios/E2.yaml --nome E2_demo --epocas 20 --imgsz 320
python ../scripts/07_contar_video.py --video dados/videos_brutos/sintetico_03.mp4 \
    --modelo runs/E2_demo/weights/best.pt --imgsz 320 --salvar-mot --rastreador ../config/bytetrack_igarapia.yaml
```

## 3. Fluxo completo

| Passo | Script | O que faz |
|---|---|---|
| 1 | `01_extrair_quadros.py` | Tira 2 quadros por segundo dos vídeos dos igarapés |
| – | (CVAT ou Label Studio) | Vocês anotam as caixas e exportam em formato YOLO |
| 2 | `02_converter_anotacoes.py` | Converte FloW, TACO e UAVVaste para o formato e as classes do projeto |
| 3 | `03_dividir_por_video.py` | Separa treino/validação/teste **por vídeo** (70/15/15) |
| 4 | `04_montar_cenarios.py` | Gera `cenarios/E1.yaml`, `E2.yaml` e `E3.yaml` |
| 5 | `05_treinar.py` | Treina o YOLO11 em cada cenário |
| 6 | `06_avaliar_deteccao.py` | P, R, mAP50 e mAP50-95 no teste, tabela comparativa |
| 7 | `07_contar_video.py` | Rastreia e conta os objetos no vídeo |
| 8 | `08_avaliar_contagem.py` | Compara a contagem automática com a manual (MAE) |
| 9 | `09_avaliar_rastreamento.py` | MOTA e IDF1 em trechos anotados com identificadores |
| – | `anonimizar_video.py` | Desfoca rostos e regiões antes de divulgar imagens |

Todos os scripts têm ajuda: `python scripts/<nome>.py -h`.

Os arquivos de `cenarios/` guardam caminhos absolutos. Se mudarem a pasta de lugar (por exemplo, do computador para o Drive), rodem o passo 4 de novo.

### Estrutura de pastas

```
dados/
  videos_brutos/          vídeos gravados (não versionar)
  quadros/<video>/        saída do passo 1
  rotulos/<video>/        .txt do YOLO exportados da ferramenta de anotação
  publicos/<base>/        saída do passo 2 (train/ e val/)
  local/                  saída do passo 3 (train/, val/, test/)
cenarios/                 E1.yaml, E2.yaml, E3.yaml
runs/                     modelos treinados (runs/<nome>/weights/best.pt)
resultados/               tabelas e vídeos anotados
```

### Comandos em sequência

```bash
python scripts/01_extrair_quadros.py --videos dados/videos_brutos --saida dados/quadros --passo 15

# (anotar no CVAT/Label Studio e salvar os .txt em dados/rotulos/<video>/)

python scripts/02_converter_anotacoes.py --formato coco --fonte taco --so-listar \
    --anotacoes bases/TACO/data/annotations.json --imagens bases/TACO/data --saida dados/publicos/taco
python scripts/02_converter_anotacoes.py --formato coco --fonte taco \
    --anotacoes bases/TACO/data/annotations.json --imagens bases/TACO/data --saida dados/publicos/taco

python scripts/03_dividir_por_video.py --quadros dados/quadros --rotulos dados/rotulos --saida dados/local
python scripts/04_montar_cenarios.py

python scripts/05_treinar.py --dados cenarios/E1.yaml --modelo yolo11n.pt --nome E1_yolo11n
python scripts/05_treinar.py --dados cenarios/E2.yaml --modelo yolo11n.pt --nome E2_yolo11n
python scripts/05_treinar.py --dados cenarios/E3.yaml --modelo yolo11n.pt --nome E3_yolo11n

python scripts/06_avaliar_deteccao.py --dados cenarios/E3.yaml --nomes E1 E2 E3 \
    --modelos runs/E1_yolo11n/weights/best.pt runs/E2_yolo11n/weights/best.pt runs/E3_yolo11n/weights/best.pt

python scripts/07_contar_video.py --video dados/videos_brutos/mindu_01.mp4 \
    --modelo runs/E2_yolo11n/weights/best.pt --salvar-mot --inicio "2026-11-10 08:30:00"
```

## 4. Classes

Definidas em `config/mapeamento_classes.yaml` (a ordem é o índice do YOLO):

0 `garrafa_plastica` · 1 `sacola_filme_plastico` · 2 `isopor` · 3 `lata_metal` · 4 `outros_residuos`

Folhas, galhos, espuma e reflexos **não** são anotados: o modelo aprende a tratá-los como fundo.
Na ferramenta de anotação, crie os rótulos exatamente com esses nomes e nessa ordem.

## 5. Bases públicas

- **FloW-Img** (Cheng et al., 2021): lixo flutuante em águas interiores. Baixe pelo site
  indicado no artigo. Rode o conversor com `--so-listar` para ver o formato e os nomes das
  classes e ajuste a seção `flow` do mapeamento.
- **TACO** (Proença e Simões, 2020): formato COCO, usa as supercategorias.
- **UAVVaste** (Kraft et al., 2021): formato COCO, imagens aéreas, uma classe genérica.

Confira a licença de cada base antes de usar.

## 6. Linha de contagem e sentido

`--linha x1 y1 x2 y2` usa coordenadas de 0 a 1. Para uma câmera na ponte olhando o
igarapé, a correnteza costuma atravessar a imagem de cima para baixo (ou o contrário),
então uma linha horizontal funciona: `--linha 0 0.5 1 0.5`.

Com `--sentido ambos` (padrão), cada objeto é contado uma vez ao cruzar a linha, em
qualquer direção. Para contar só o que desce a correnteza, rode uma vez, veja a coluna
`sentido` em `contagens.csv` e depois use `--sentido positivo` ou `--sentido negativo`.

O rastreador está configurado em `config/bytetrack_igarapia.yaml`. Se objetos parcialmente
submersos perderem o identificador, aumente `track_buffer` (quantos quadros um objeto
pode sumir antes de ser esquecido).

## 7. Licença

O Ultralytics YOLO usa a licença AGPL-3.0; por isso este código também deve ser
distribuído como código aberto sob AGPL-3.0.
