# Planta baixa automática a partir de nuvem de pontos do SLAM 100

Extrai a planta baixa da Seção de Engenharia Cartográfica do IME — paredes com
espessura, portas e janelas — de uma nuvem registrada, e grava em **DXF** por
camadas. Entrada: o `.las`/`.laz` exportado pelo SLAM GO POST. Saída: um desenho
em metros que abre no AutoCAD, QGIS ou CloudCompare.

## Instalação

O núcleo usa só `numpy`, `scipy`, `laspy[lazrs]`, `ezdxf` e `matplotlib` — todos
com wheel para Python 3.13, então **não depende de `open3d`** (que ainda não
publica wheel para 3.13). O `open3d` só é necessário para a visualização 3D
opcional do notebook, e aí vale o ambiente 3.12 descrito no README do curso.

```powershell
git clone https://github.com/caioalvesvb/IPE2-SLAM-100.git
cd IPE2-SLAM-100
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

> `numpy>=2.0` não é preferência: a máscara de parede conta bits de ocupação
> vertical com `np.bitwise_count`, que só existe a partir dessa versão.

Os dados do levantamento não estão versionados aqui — o `.pcap` de um projeto
SLAM 100 passa de 900 MB, acima do limite do GitHub. O repositório guarda o
algoritmo; aponte a CLI para a pasta onde a nuvem estiver.

## Uso

```powershell
python .\planta_baixa.py ..\SN_000055_certo -o planta_secao_cartografica.dxf --diagnosticos diag\
```

A entrada aceita a pasta do projeto — varrida em busca do `.las`/`.laz`, com
preferência pelos nomes que o SLAM GO POST usa (`result`, `filtered`) — ou o
arquivo direto.

Uso programático:

```python
from plantabaixa import gerar_planta
resultado = gerar_planta("../SN_000055_certo", saida="planta.dxf", diagnosticos="diag")
print(resultado.resumo())
```

## Como funciona

Seis etapas, cada uma num módulo testável isoladamente.

**1. Carga e nivelamento** (`plantabaixa/nuvem.py`). A nuvem é lida em blocos e
reduzida a voxels de 2 cm na leitura, o que mantém a memória sob controle num
export de 13 minutos. Em seguida é nivelada: o piso é ajustado por RANSAC no
quartil inferior de altura — e não pelo pico do histograma, que se esborracha
quando a nuvem chega desaprumada —, a nuvem é girada até o piso ficar horizontal
e transladada para o piso ficar em `z = 0`. O nivelamento não é opcional: 1 grau
de desaprumo desloca o teto em 50 cm numa sala de 30 m.

**2. Máscara de parede por persistência vertical** (`plantabaixa/paredes.py`).
Esta é a decisão central. Cortar a nuvem numa faixa fixa de altura é o caminho
óbvio e é também o que falha em ambiente mobiliado: no corte de 1,0 a 2,0 m um
armário de 2 m é idêntico a uma parede. O que separa os dois não é onde estão, e
sim **quanto do vão piso-teto cada coluna ocupa** e **se ela encosta no teto**.
Parede atravessa o vão inteiro; armário para antes. A parede escondida atrás do
armário fica de fora da máscara — ela está ocluída —, vira lacuna e é resolvida
na etapa 4.

**3. Vetorização** (`plantabaixa/paredes.py`). As células são classificadas pela
orientação local via tensor de estrutura, o que entrega a cada eixo apenas as
células que lhe pertencem. Sem isso, o histograma de um eixo recebe como fundo
as células de todas as paredes perpendiculares a ele e fabrica picos onde não há
parede. Cada eixo então acumula uma densidade suavizada dos deslocamentos, e
cada pico é uma face. Faces paralelas a 8–35 cm com extensões sobrepostas são os
dois lados de uma parede, e a distância entre elas é a espessura medida. As
pontas são estendidas até a face distante da parede perpendicular, fechando os
cantos. Paredes fora de esquadro saem por RANSAC 2D no resto das células.

**4. Portas e janelas** (`plantabaixa/vaos.py`). Procurar vão como lacuna na
máscara 2D não funciona: a coluna sobre uma janela tem alvenaria embaixo e em
cima, o que dá persistência em torno de 0,5 — exatamente o limiar que separa
parede de móvel. O lugar sem ambiguidade é o plano da própria parede: projetando
os pontos em `(s, h)`, a alvenaria é região cheia e cada abertura é um retângulo
vazio. A classificação sai da geometria do retângulo mais uma evidência
independente: os pontos que caem **dentro da pegada** da parede. Dentro da
espessura de alvenaria maciça não há nada para o sensor ver; se existem pontos
ali, é porque há abertura, e a altura deles diz qual — piso atravessando junto ao
chão é porta, peitoril entre 0,4 e 1,5 m é janela, nada é oclusão. As bordas são
depois acertadas nos pontos de ombreira, peitoril e verga, não na grade.

**5. DXF** (`plantabaixa/dxf.py`). O corte horizontal passa a 1,40 m e só
interrompe a alvenaria nas aberturas que ele atravessa — um basculante alto é
representado tracejado sem seccionar a parede, como um projetista faria. As
paredes saem como polilinhas fechadas, uma por trecho entre aberturas, para que
o vão seja lacuna de verdade no CAD.

**6. Controle de qualidade** (`plantabaixa/qc.py`). `relatorio_planta.json` e
`.txt` com resíduo por parede, espessuras, vãos e avisos; e cinco famílias de
figura, entre elas o mapa `(s,h)` de cada parede, que mostra exatamente onde a
etapa 4 decidiu.

### Camadas do DXF

| Camada | Conteúdo |
|---|---|
| `PAREDES` | Corpo da alvenaria, polilinha fechada por trecho |
| `EIXOS` | Eixo de cada parede |
| `PORTAS` | Ombreiras, folha e arco de abertura |
| `JANELAS` | Peitoril nas duas faces e ombreiras |
| `VAOS_ALTOS` | Aberturas acima do plano de corte |
| `AVISOS` | Trechos ocluídos, parede de face única, suspeita de porta fechada |
| `TEXTO` | Rótulos dos vãos e notas do levantamento |

## Parâmetros

Os valores padrão são ponto de partida para interior de alvenaria, não valores
finais. O relatório traz os números que orientam o ajuste de cada um.

| Parâmetro | Padrão | Para que serve |
|---|---|---|
| `--voxel` | 0,02 m | Redução da nuvem na leitura |
| `--res` | 0,02 m | Resolução da grade 2D. Precisa caber ~3 células na espessura da parede mais fina; o relatório avisa quando não cabe |
| `--persistencia` | 0,50 | Fração do vão piso-teto para a célula valer como parede. Baixe se houver muita oclusão; suba se móvel alto estiver virando parede |
| `--min-parede` | 0,40 m | Comprimento mínimo de face |
| `--lacuna` | 0,30 m | Vazio que conta como lacuna, em vez de ser costurado |
| `--espessura-max` | 0,35 m | Maior separação aceita entre as duas faces de uma parede |
| `--espessura-padrao` | 0,15 m | Adotada em parede vista de um lado só |
| `--tol-angulo` | 12° | Desvio tolerado para a célula pertencer a um eixo |
| `--sem-manhattan` | — | Desliga a regularização por eixos, para prédio não retilíneo |
| `--altura-corte` | 1,40 m | Plano de corte da planta |
| `--peitoril` | 0,40 1,55 m | Faixa de peitoril que caracteriza janela |
| `--largura-porta` | 0,55 1,70 m | Faixa de largura de porta; acima vira passagem |
| `--nucleo-minimo` | 15 | Pontos dentro da espessura para o vão não ser oclusão |
| `--recorte` | — | `x0,y0,x1,y1` nas coordenadas da nuvem de entrada |

## Limitações conhecidas

- **Porta fechada no momento do levantamento é indetectável.** A folha é uma
  superfície plana e lê como alvenaria. O que o algoritmo faz é reportar como
  *suspeita* o trecho de face recuado de 2 a 9 cm com largura de porta, na camada
  `AVISOS`. É indício, nunca detecção.
- **Trecho sem visada vira oclusão, não abertura.** É o comportamento correto, e
  a parede é desenhada contínua por cima dele — mas aquele trecho não foi medido,
  e isso consta do relatório.
- **O lado para o qual a porta abre é hipótese de desenho.** A nuvem não carrega
  essa informação; adota-se o lado com mais piso levantado.
- **Um pavimento por execução.** Massa de pontos acima do teto é recortada e
  avisada. Para outro nível, rode de novo com `--recorte`.
- **A regularização Manhattan supõe prédio retilíneo.** Paredes fora de esquadro
  saem pelo passe oblíquo e são sinalizadas; `--sem-manhattan` desliga tudo.

## Testes

```powershell
python -m pytest tests\ -v
```

A verificação não depende da nuvem real. `tests/sintetico.py` gera uma nuvem de
planta conhecida — 8 paredes, 3 portas, 2 janelas, móveis que encostam na parede,
dois trechos ocluídos, ruído de 1 cm, desaprumo e translação em coordenadas de
projeto — e `tests/test_sintetico.py` exige que o pipeline recupere essa
geometria dentro de 2,5 cm nas paredes e 8 cm nos vãos, com a classe certa e sem
inventar abertura. Inclui os casos que mais importam para dados reais: nuvem
girada em azimute, resolução alternativa, móvel alto que não pode virar parede e
oclusão que não pode virar porta.
