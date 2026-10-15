"""Etapa 4: portas e janelas, pelo mapa (s, h) de cada parede.

Detectar vao como lacuna na mascara 2D nao funciona, e vale registrar por que: a
coluna vertical sobre uma janela tem alvenaria embaixo (do piso ao peitoril) e em
cima (da verga ao teto), o que da uma persistencia vertical em torno de 0,5 -
exatamente o limiar que separa parede de movel. A janela cairia de um lado ou do
outro ao sabor de um centimetro de peitoril.

O lugar sem ambiguidade e o plano da propria parede. Projetando os pontos da
parede em `(s, h)` - abscissa ao longo do eixo e altura -, a alvenaria aparece
como regiao cheia e cada abertura como um retangulo vazio. A classificacao sai da
geometria do retangulo e de uma evidencia independente: os pontos que caem
*dentro da pegada* da parede.

Essa evidencia e o que separa abertura de oclusao, e vem da fisica do
levantamento. Dentro da espessura de uma parede macica nao ha nada para o sensor
ver. Se existem pontos ali, e porque ha abertura, e a altura deles diz qual:

- piso atravessando a espessura, com base junto ao chao -> porta ou passagem;
- peitoril entre 0,4 m e 1,5 m -> janela;
- nada dentro da pegada -> o trecho nao foi visto, e oclusao, e a parede deve
  ser desenhada continua por cima dela.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from plantabaixa.paredes import Parede

CLASSES_ABERTAS = ("porta", "passagem", "janela")


@dataclass
class Vao:
    """Abertura detectada numa parede, em coordenadas do referencial do predio."""

    parede: str
    classe: str
    inicio: float
    fim: float
    base: float
    topo: float
    pontos_nucleo: int = 0
    preenchimento: float = 0.0
    lado: int = 1
    observacao: str = ""

    @property
    def largura(self) -> float:
        return self.fim - self.inicio

    @property
    def altura(self) -> float:
        return self.topo - self.base

    @property
    def centro(self) -> float:
        return 0.5 * (self.inicio + self.fim)

    @property
    def aberto(self) -> bool:
        """Se o vao interrompe a alvenaria e portanto quebra a parede no desenho."""
        return self.classe in CLASSES_ABERTAS


@dataclass
class Limites:
    """Faixas geometricas que definem cada classe de vao, todas em metros."""

    base_porta: float = 0.18
    altura_minima_porta: float = 1.60
    largura_minima_porta: float = 0.55
    largura_maxima_porta: float = 1.70
    peitoril_minimo: float = 0.40
    peitoril_maximo: float = 1.55
    altura_minima_janela: float = 0.40
    largura_minima_janela: float = 0.40
    largura_minima_passagem: float = 1.70
    preenchimento_minimo: float = 0.55
    pontos_nucleo_minimos: int = 15
    celula: float = 0.05
    tolerancia_face: float = 0.045
    margem_nucleo: float = 0.030
    recuo_porta_fechada: tuple[float, float] = (0.02, 0.09)


@dataclass
class Analise:
    """Vaos de uma parede e o material de diagnostico do mapa (s, h)."""

    vaos: list[Vao]
    material: np.ndarray
    s0: float
    h0: float
    celula: float
    avisos: list[str] = field(default_factory=list)


def _coordenadas_da_parede(parede: Parede, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distancia ao eixo da parede e abscissa ao longo dela."""
    d = xy @ parede.normal - parede.eixo_central
    s = xy @ parede.direcao
    return d, s


def _faixa_vertical_util(material: np.ndarray, fracao: float = 0.5) -> tuple[int, int]:
    """Primeira e ultima linha de altura em que a parede tem material de fato.

    Sem este recorte, a faixa vazia acima do topo da parede atravessa o mapa de
    ponta a ponta e costura todas as aberturas numa unica componente conexa.
    """
    cobertura = material.mean(axis=0)
    presentes = np.flatnonzero(cobertura >= fracao)
    if not len(presentes):
        presentes = np.flatnonzero(cobertura > 0)
    if not len(presentes):
        return 0, material.shape[1] - 1
    return int(presentes[0]), int(presentes[-1])


def _altura_base_nucleo(alturas: np.ndarray, celula: float, fracao: float = 0.12) -> tuple[float, float]:
    """Menor e maior altura com massa relevante no nucleo da parede.

    O nucleo de uma porta reune o piso que atravessa a espessura (muitos pontos
    junto ao chao), as ombreiras (poucos pontos, espalhados na vertical) e a
    verga (muitos pontos no alto). Tomar o minimo e o maximo brutos pegaria
    ruido; exigir uma fracao do bin mais cheio isola as superficies reais.
    """
    if not len(alturas):
        return float("nan"), float("nan")
    n_bins = max(2, int(np.ceil((alturas.max() - alturas.min()) / celula)) + 1)
    hist, bordas = np.histogram(alturas, bins=n_bins)
    centros = 0.5 * (bordas[:-1] + bordas[1:])
    relevantes = np.flatnonzero(hist >= max(2.0, hist.max() * fracao))
    if not len(relevantes):
        return float(alturas.min()), float(alturas.max())
    return float(centros[relevantes[0]]), float(centros[relevantes[-1]])


def _refinar(
    inicio: float,
    fim: float,
    base: float,
    topo: float,
    s_perto: np.ndarray,
    h_perto: np.ndarray,
    celula: float,
) -> tuple[float, float, float, float]:
    """Acerta as bordas do vao nos pontos de alvenaria, nao na grade.

    A componente conexa devolve o retangulo vazio em multiplos da celula, o que
    tira sistematicamente meia celula de cada lado. Como a cota do vao e produto
    de entrega, as quatro bordas sao reposicionadas na ombreira, no peitoril e na
    verga reais - os pontos de alvenaria imediatamente vizinhos ao vazio.
    """
    busca = 3 * celula
    meia_altura = 0.5 * (base + topo)
    meio = 0.5 * (inicio + fim)

    # Ombreiras: material na faixa de altura do vao, logo fora dele.
    faixa = (h_perto > base + celula) & (h_perto < topo - celula)
    esquerda = faixa & (s_perto >= inicio - busca) & (s_perto <= meio)
    direita = faixa & (s_perto <= fim + busca) & (s_perto >= meio)
    if esquerda.any():
        inicio = float(s_perto[esquerda].max())
    if direita.any():
        fim = float(s_perto[direita].min())

    # Peitoril e verga: material na largura do vao, logo abaixo e logo acima.
    dentro = (s_perto > inicio + celula) & (s_perto < fim - celula)
    abaixo = dentro & (h_perto >= base - busca) & (h_perto <= meia_altura)
    acima = dentro & (h_perto <= topo + busca) & (h_perto >= meia_altura)
    if abaixo.any():
        base = float(h_perto[abaixo].max())
    if acima.any():
        topo = float(h_perto[acima].min())
    return inicio, fim, base, topo


def _lado_do_ambiente(
    d: np.ndarray, s: np.ndarray, z: np.ndarray, inicio: float, fim: float, meia: float
) -> int:
    """+1 ou -1: lado do vao com mais piso visivel, onde a folha sera desenhada.

    Nao ha como saber pela nuvem para que lado uma porta abre; a convencao
    adotada e desenhar a folha no lado com mais piso levantado, e isso fica
    registrado no relatorio como hipotese de desenho.
    """
    vizinho = (s >= inicio - 1.5) & (s <= fim + 1.5) & (z <= 0.15)
    contagens = []
    for lado in (1.0, -1.0):
        faixa = vizinho & (lado * d > meia) & (lado * d <= meia + 1.5)
        contagens.append(int(faixa.sum()))
    return 1 if contagens[0] >= contagens[1] else -1


def _classificar(
    largura: float,
    base: float,
    topo: float,
    preenchimento: float,
    nucleo: int,
    base_nucleo: float,
    limites: Limites,
) -> tuple[str, str]:
    """Decide a classe do retangulo vazio e devolve (classe, observacao)."""
    altura = topo - base
    if nucleo < limites.pontos_nucleo_minimos:
        return "oclusao", "sem pontos dentro da espessura: trecho nao levantado"
    if preenchimento < limites.preenchimento_minimo:
        return "oclusao", f"regiao vazia pouco retangular (preenchimento {preenchimento:.2f})"

    junto_ao_piso = base <= limites.base_porta and (
        np.isnan(base_nucleo) or base_nucleo <= limites.base_porta + 0.10
    )
    if junto_ao_piso and altura >= limites.altura_minima_porta:
        if largura >= limites.largura_minima_passagem:
            return "passagem", ""
        if largura >= limites.largura_minima_porta:
            return "porta", ""
        return "indeterminado", f"vao junto ao piso com apenas {largura:.2f} m de largura"

    peitoril = base if np.isnan(base_nucleo) else max(base, base_nucleo)
    if (
        limites.peitoril_minimo <= peitoril <= limites.peitoril_maximo
        and altura >= limites.altura_minima_janela
        and largura >= limites.largura_minima_janela
    ):
        return "janela", ""
    return "indeterminado", f"base {base:.2f} m e altura {altura:.2f} m fora das faixas"


def analisar_parede(
    parede: Parede,
    xyz: np.ndarray,
    z_teto: float,
    limites: Limites | None = None,
) -> Analise:
    """Encontra e classifica os vaos de uma parede."""
    from scipy.ndimage import binary_closing, label

    limites = limites or Limites()
    celula = limites.celula
    d, s = _coordenadas_da_parede(parede, xyz[:, :2])
    meia = parede.espessura / 2

    perto = (
        (np.abs(d) <= meia + limites.tolerancia_face)
        & (s >= parede.s_min - celula)
        & (s <= parede.s_max + celula)
    )
    if perto.sum() < 50:
        return Analise([], np.zeros((1, 1), bool), parede.s_min, 0.0, celula,
                        [f"{parede.nome}: pontos insuficientes para analisar vaos."])

    s_parede, h_parede = s[perto], xyz[perto, 2]
    s0, h0 = parede.s_min, 0.0
    ns = max(2, int(np.ceil((parede.s_max - s0) / celula)) + 1)
    nh = max(2, int(np.ceil((z_teto - h0) / celula)) + 1)

    material = np.zeros((ns, nh), dtype=bool)
    is_ = np.clip(((s_parede - s0) / celula).astype(np.int64), 0, ns - 1)
    ih = np.clip(((h_parede - h0) / celula).astype(np.int64), 0, nh - 1)
    material[is_, ih] = True
    # Costura falhas de amostragem de uma celula sem fechar abertura real: a
    # porta mais estreita tem 0,55 m, ou 11 celulas de 5 cm.
    material = binary_closing(material, structure=np.ones((3, 3), bool), iterations=1)

    ih0, ih1 = _faixa_vertical_util(material)
    janela = material[:, ih0 : ih1 + 1]
    vazio = ~janela
    rotulos, n = label(vazio)  # conectividade 4: nao costura aberturas na diagonal

    # Nucleo: pontos estritamente dentro da pegada da parede. A margem evita que
    # o ruido das faces vaze para dentro; em parede fina usa-se metade da meia
    # espessura, que e o maximo disponivel.
    margem = min(limites.margem_nucleo, meia / 2)
    no_nucleo = np.abs(d) <= meia - margem

    vaos: list[Vao] = []
    avisos: list[str] = []
    for rotulo in range(1, n + 1):
        celulas = rotulos == rotulo
        idx_s, idx_h = np.nonzero(celulas)
        inicio = s0 + idx_s.min() * celula
        fim = s0 + (idx_s.max() + 1) * celula
        base = h0 + (ih0 + idx_h.min()) * celula
        topo = h0 + (ih0 + idx_h.max() + 1) * celula
        largura, altura = fim - inicio, topo - base
        if largura < limites.largura_minima_janela or altura < limites.altura_minima_janela:
            continue
        preenchimento = float(celulas.sum()) / max(1, len(np.unique(idx_s)) * len(np.unique(idx_h)))

        inicio, fim, base, topo = _refinar(
            inicio, fim, base, topo, s_parede, h_parede, celula
        )
        largura, altura = fim - inicio, topo - base
        if largura < limites.largura_minima_janela or altura < limites.altura_minima_janela:
            continue

        na_faixa = no_nucleo & (s >= inicio) & (s <= fim)
        nucleo = int(na_faixa.sum())
        base_nucleo, _ = _altura_base_nucleo(xyz[na_faixa, 2], celula)

        classe, observacao = _classificar(
            largura, base, topo, preenchimento, nucleo, base_nucleo, limites
        )
        vaos.append(
            Vao(
                parede=parede.nome,
                classe=classe,
                inicio=inicio,
                fim=fim,
                base=base,
                topo=topo,
                pontos_nucleo=nucleo,
                preenchimento=preenchimento,
                lado=_lado_do_ambiente(d, s, xyz[:, 2], inicio, fim, meia),
                observacao=observacao,
            )
        )

    for suspeita, recuo in _portas_fechadas(
        parede, d, s, xyz[:, 2], material, s0, celula, limites
    ):
        vaos.append(suspeita)
        avisos.append(
            f"{parede.nome}: possivel porta fechada em s = {suspeita.centro:.2f} m "
            f"({suspeita.largura:.2f} m de largura, face recuada {recuo * 100:.1f} cm). "
            "Indicio, nao deteccao: confira no levantamento."
        )

    vaos.sort(key=lambda v: v.inicio)
    return Analise(vaos, material, s0, h0, celula, avisos)


def _portas_fechadas(
    parede: Parede,
    d: np.ndarray,
    s: np.ndarray,
    z: np.ndarray,
    material: np.ndarray,
    s0: float,
    celula: float,
    limites: Limites,
) -> list[tuple[Vao, float]]:
    """Pista de porta fechada: trecho de face recuado em poucos centimetros.

    Uma porta fechada na hora do levantamento e indetectavel como vao - a folha
    e uma superficie plana e le como alvenaria. O que resta e um indicio: a folha
    costuma ficar alguns centimetros recuada em relacao ao plano da face. Isto e
    reportado como suspeita, nunca como deteccao, e nao quebra a parede.
    """
    meia = parede.espessura / 2
    recuo_min, recuo_max = limites.recuo_porta_fechada
    suspeitas: list[tuple[Vao, float]] = []

    for lado in (1.0, -1.0):
        na_face = (
            (lado * d > meia * 0.3)
            & (np.abs(d) <= meia + limites.tolerancia_face)
            & (z >= 0.3)
            & (z <= limites.altura_minima_porta)
        )
        if na_face.sum() < 200:
            continue
        coluna = np.clip(((s[na_face] - s0) / celula).astype(np.int64), 0, material.shape[0] - 1)
        profundidade = lado * d[na_face]
        soma = np.bincount(coluna, weights=profundidade, minlength=material.shape[0])
        contagem = np.bincount(coluna, minlength=material.shape[0])
        valida = contagem >= 8
        if valida.sum() < 10:
            continue
        media = np.divide(soma, contagem, out=np.full_like(soma, np.nan), where=valida)
        referencia = float(np.nanpercentile(media, 85))
        recuo = referencia - media
        recuada = valida & (recuo >= recuo_min) & (recuo <= recuo_max)

        # Corridas contiguas de colunas recuadas com largura de porta.
        bordas = np.diff(np.concatenate([[False], recuada, [False]]).astype(np.int8))
        for i, f in zip(np.flatnonzero(bordas == 1), np.flatnonzero(bordas == -1)):
            largura = (f - i) * celula
            if not limites.largura_minima_porta <= largura <= limites.largura_maxima_porta:
                continue
            medio = float(np.nanmean(recuo[i:f]))
            suspeitas.append(
                (
                    Vao(
                        parede=parede.nome,
                        classe="porta_fechada?",
                        inicio=s0 + i * celula,
                        fim=s0 + f * celula,
                        base=0.0,
                        topo=limites.altura_minima_porta,
                        pontos_nucleo=0,
                        lado=int(lado),
                        observacao=f"face recuada {medio * 100:.1f} cm; indicio, nao deteccao",
                    ),
                    medio,
                )
            )
    return suspeitas


def detectar_vaos(
    paredes: list[Parede],
    xyz: np.ndarray,
    z_teto: float,
    limites: Limites | None = None,
) -> tuple[list[Vao], dict[str, Analise], list[str]]:
    """Roda a analise em todas as paredes e devolve vaos, diagnosticos e avisos."""
    limites = limites or Limites()
    todos: list[Vao] = []
    analises: dict[str, Analise] = {}
    avisos: list[str] = []
    for parede in paredes:
        analise = analisar_parede(parede, xyz, z_teto, limites)
        analises[parede.nome] = analise
        todos.extend(analise.vaos)
        avisos.extend(analise.avisos)
    return todos, analises, avisos
