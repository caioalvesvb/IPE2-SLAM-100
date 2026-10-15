"""Gerador de nuvem sintetica com planta baixa conhecida.

Serve de verdade de referencia: as paredes, portas e janelas sao definidas aqui
em metros e o teste compara o que o pipeline recupera com estes numeros. A
geometria imita o que um levantamento SLAM 100 de interior produz, incluindo o
que mais atrapalha a extracao automatica:

- paredes de alvenaria com duas faces e espessura real;
- moveis que grudam na parede (mesas a 0,75 m, armarios a 2,0 m), que um corte
  por faixa de altura confundiria com parede;
- oclusao: faixas de parede sem nenhum ponto, atras dos armarios;
- ruido gaussiano na normal de cada superficie;
- desaprumo e translacao, para exercitar o nivelamento.

Planta (eixos em metros, origem no canto inferior esquerdo do perimetro):

    (0,10) +---------------------+---------------+ (16,10)
           |                     |               |
           |      SALA_A         |    SALA_B     |
           |                     |               |
    (0,6)  +---J1-------+--P2----+------J2-------+
           |            |                        |
           |  CORREDOR  P1                       |
           |            |        SALA_C          |
    (0,3)  +------------+                        |
           |                                     |
           |                                     |
    (0,0)  +-------------P3----------------------+ (16,0)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Dimensoes do predio sintetico.
LARGURA = 16.0
PROFUNDIDADE = 10.0
ALTURA = 2.80
ESPESSURA = 0.15
ESPESSURA_INTERNA = 0.12


@dataclass(frozen=True)
class Vao:
    """Abertura num segmento de parede, em coordenadas do eixo da parede."""

    parede: str
    classe: str
    centro: float
    largura: float
    base: float
    altura: float

    @property
    def inicio(self) -> float:
        return self.centro - self.largura / 2

    @property
    def fim(self) -> float:
        return self.centro + self.largura / 2


@dataclass(frozen=True)
class ParedeVerdade:
    """Parede do gabarito: eixo, extensao e espessura conhecidos."""

    nome: str
    eixo: str          # "x" para parede que corre em x, "y" para parede que corre em y
    offset: float      # posicao do eixo da parede no eixo perpendicular
    inicio: float
    fim: float
    espessura: float


# --- Gabarito -------------------------------------------------------------

PAREDES = [
    # Perimetro.
    ParedeVerdade("SUL", "x", 0.0, 0.0, LARGURA, ESPESSURA),
    ParedeVerdade("NORTE", "x", PROFUNDIDADE, 0.0, LARGURA, ESPESSURA),
    ParedeVerdade("OESTE", "y", 0.0, 0.0, PROFUNDIDADE, ESPESSURA),
    ParedeVerdade("LESTE", "y", LARGURA, 0.0, PROFUNDIDADE, ESPESSURA),
    # Divisorias internas.
    ParedeVerdade("DIV_6", "x", 6.0, 0.0, LARGURA, ESPESSURA_INTERNA),
    ParedeVerdade("DIV_3", "x", 3.0, 0.0, 6.0, ESPESSURA_INTERNA),
    ParedeVerdade("DIV_SALAS", "y", 9.0, 6.0, PROFUNDIDADE, ESPESSURA_INTERNA),
    ParedeVerdade("DIV_CORR", "y", 6.0, 3.0, 6.0, ESPESSURA_INTERNA),
]

VAOS = [
    Vao("DIV_CORR", "porta", centro=4.5, largura=0.90, base=0.0, altura=2.10),
    Vao("DIV_6", "porta", centro=11.0, largura=1.00, base=0.0, altura=2.10),
    Vao("SUL", "porta", centro=8.0, largura=1.20, base=0.0, altura=2.20),
    Vao("DIV_6", "janela", centro=2.5, largura=1.40, base=1.00, altura=1.20),
    Vao("LESTE", "janela", centro=8.0, largura=1.60, base=0.95, altura=1.30),
]

# Moveis: (x0, y0, x1, y1, altura). Os armarios encostam em paredes de proposito.
MOVEIS = [
    (0.60, 6.40, 2.20, 8.60, 2.00),   # armario alto na SALA_A, contra DIV_6
    (10.0, 9.10, 13.0, 9.70, 2.00),   # armario alto na SALA_B, contra NORTE
    (3.0, 7.0, 5.4, 8.2, 0.75),       # mesas
    (11.0, 2.0, 14.0, 4.0, 0.75),
    (7.0, 1.0, 9.0, 2.2, 1.10),       # divisoria baixa de baia
]

# Faixas de parede deixadas sem pontos, simulando oclusao total.
OCLUSOES = [
    ("NORTE", 1.0, 3.0),
    ("OESTE", 7.0, 8.5),
]

# Porta fechada no momento do levantamento: a folha preenche o vao, recuada em
# relacao ao plano da face. Opcional, porque a nuvem padrao representa o
# levantamento bem-comportado; e usada no teste da heuristica de suspeita.
PORTA_FECHADA = Vao("DIV_6", "porta", centro=14.0, largura=0.90, base=0.0, altura=2.10)
RECUO_FOLHA = 0.040
ESPESSURA_FOLHA = 0.035


def _amostrar_retangulo(
    canto: np.ndarray, u: np.ndarray, v: np.ndarray, du: float, dv: float, passo: float
) -> np.ndarray:
    """Amostra uma face retangular plana com espacamento aproximado `passo`."""
    nu = max(2, int(round(du / passo)))
    nv = max(2, int(round(dv / passo)))
    su = (np.arange(nu) + 0.5) / nu * du
    sv = (np.arange(nv) + 0.5) / nv * dv
    malha_u, malha_v = np.meshgrid(su, sv, indexing="ij")
    return canto + malha_u.reshape(-1, 1) * u + malha_v.reshape(-1, 1) * v


def _faces_da_parede(parede: ParedeVerdade) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Duas faces da parede: (canto, direcao_ao_longo, normal_da_face)."""
    meia = parede.espessura / 2
    if parede.eixo == "x":
        ao_longo = np.array([1.0, 0.0, 0.0])
        return [
            (np.array([parede.inicio, parede.offset - meia, 0.0]), ao_longo, np.array([0.0, -1.0, 0.0])),
            (np.array([parede.inicio, parede.offset + meia, 0.0]), ao_longo, np.array([0.0, 1.0, 0.0])),
        ]
    ao_longo = np.array([0.0, 1.0, 0.0])
    return [
        (np.array([parede.offset - meia, parede.inicio, 0.0]), ao_longo, np.array([-1.0, 0.0, 0.0])),
        (np.array([parede.offset + meia, parede.inicio, 0.0]), ao_longo, np.array([1.0, 0.0, 0.0])),
    ]


def _mascara_vaos(
    parede: str, s: np.ndarray, z: np.ndarray, extras: tuple[Vao, ...] = ()
) -> np.ndarray:
    """True onde o ponto cai dentro de um vao e portanto nao deve existir."""
    remover = np.zeros(len(s), dtype=bool)
    for vao in tuple(VAOS) + extras:
        if vao.parede != parede:
            continue
        remover |= (
            (s >= vao.inicio)
            & (s <= vao.fim)
            & (z >= vao.base)
            & (z <= vao.base + vao.altura)
        )
    return remover


def _mascara_oclusao(parede: str, s: np.ndarray) -> np.ndarray:
    """True onde a parede ficou sem visada no levantamento."""
    remover = np.zeros(len(s), dtype=bool)
    for nome, s0, s1 in OCLUSOES:
        if nome == parede:
            remover |= (s >= s0) & (s <= s1)
    return remover


def _faixa_da_parede(parede: ParedeVerdade, xy: np.ndarray) -> np.ndarray:
    """True onde o ponto cai na pegada (projecao) do corpo da parede."""
    meia = parede.espessura / 2
    perpendicular, ao_longo = (1, 0) if parede.eixo == "x" else (0, 1)
    return (
        (np.abs(xy[:, perpendicular] - parede.offset) <= meia)
        & (xy[:, ao_longo] >= parede.inicio)
        & (xy[:, ao_longo] <= parede.fim)
    )


def _dentro_de_parede(xy: np.ndarray, vaos_passantes: bool) -> np.ndarray:
    """True onde ha alvenaria. `vaos_passantes` abre a pegada nos vaos de porta."""
    dentro = np.zeros(len(xy), dtype=bool)
    for parede in PAREDES:
        faixa = _faixa_da_parede(parede, xy)
        if vaos_passantes:
            ao_longo = 0 if parede.eixo == "x" else 1
            for vao in VAOS:
                # Porta e passagem tem piso continuo; janela nao.
                if vao.parede != parede.nome or vao.base > 0.15:
                    continue
                faixa &= ~(
                    (xy[:, ao_longo] >= vao.inicio) & (xy[:, ao_longo] <= vao.fim)
                )
        dentro |= faixa
    return dentro


def _superficies_dos_vaos(passo: float) -> list[np.ndarray]:
    """Peitoril, verga e ombreiras de cada vao, atravessando a espessura.

    Sao as unicas superficies que o sensor ve dentro da pegada de uma parede, e
    por isso a evidencia de que ali existe abertura e nao oclusao.
    """
    por_nome = {p.nome: p for p in PAREDES}
    partes: list[np.ndarray] = []
    for vao in VAOS:
        parede = por_nome[vao.parede]
        espessura = parede.espessura
        if parede.eixo == "x":
            # Parede corre em x: a espessura esta em y.
            def ponto(s: float, t: float, h: float) -> np.ndarray:
                return np.array([s, parede.offset - espessura / 2 + t, h])

            u_largura = np.array([1.0, 0.0, 0.0])
            u_espessura = np.array([0.0, 1.0, 0.0])
        else:
            def ponto(s: float, t: float, h: float) -> np.ndarray:
                return np.array([parede.offset - espessura / 2 + t, s, h])

            u_largura = np.array([0.0, 1.0, 0.0])
            u_espessura = np.array([1.0, 0.0, 0.0])

        # Peitoril (so em janela: em porta o piso ja atravessa) e verga.
        alturas = [vao.base + vao.altura]
        if vao.base > 0.15:
            alturas.append(vao.base)
        for h in alturas:
            partes.append(
                _amostrar_retangulo(
                    ponto(vao.inicio, 0.0, h), u_largura, u_espessura, vao.largura, espessura, passo
                )
            )
        # Ombreiras: as duas faces laterais do vao.
        for s in (vao.inicio, vao.fim):
            partes.append(
                _amostrar_retangulo(
                    ponto(s, 0.0, vao.base),
                    u_espessura,
                    np.array([0.0, 0.0, 1.0]),
                    espessura,
                    vao.altura,
                    passo,
                )
            )
    return partes


def _folha_de_porta_fechada(passo: float) -> list[np.ndarray]:
    """Folha de porta fechada, recuada em relacao ao plano da face.

    Uma porta fechada nao e vao: a folha preenche a abertura e o sensor so ve
    uma superficie plana. O unico vestigio e o recuo de poucos centimetros em
    relacao a face da parede, que e o que a heuristica de suspeita procura.
    """
    parede = next(p for p in PAREDES if p.nome == PORTA_FECHADA.parede)
    meia = parede.espessura / 2
    frente = meia - RECUO_FOLHA
    fundo = frente - ESPESSURA_FOLHA
    partes: list[np.ndarray] = []
    for profundidade in (frente, fundo):
        canto = np.array([PORTA_FECHADA.inicio, parede.offset + profundidade, 0.0])
        partes.append(
            _amostrar_retangulo(
                canto, np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]),
                PORTA_FECHADA.largura, PORTA_FECHADA.altura, passo,
            )
        )
    # Batentes: do plano de cada face ate a folha.
    for s in (PORTA_FECHADA.inicio, PORTA_FECHADA.fim):
        canto = np.array([s, parede.offset - meia, 0.0])
        partes.append(
            _amostrar_retangulo(
                canto, np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]),
                parede.espessura, PORTA_FECHADA.altura, passo,
            )
        )
    return partes


def gerar(
    passo: float = 0.03,
    ruido: float = 0.010,
    desaprumo_graus: tuple[float, float] = (1.5, -0.8),
    azimute_graus: float = 0.0,
    translacao: tuple[float, float, float] = (712345.6, 7458765.4, 31.7),
    porta_fechada: bool = False,
    semente: int = 55,
) -> np.ndarray:
    """Gera a nuvem sintetica completa, ja desaprumada, girada e transladada.

    O desaprumo e a translacao imitam uma nuvem exportada em coordenadas de
    projeto: o pipeline precisa nivelar e trazer o piso para zero por conta
    propria, sem receber nenhuma dica. `azimute_graus` gira o predio no plano
    horizontal, situacao normal numa nuvem real e que obriga a estimativa de
    direcao dominante a funcionar.
    """
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(semente)
    partes: list[np.ndarray] = []
    extras = (PORTA_FECHADA,) if porta_fechada else ()

    # Paredes: duas faces cada, com vaos e oclusoes recortados.
    for parede in PAREDES:
        comprimento = parede.fim - parede.inicio
        for canto, ao_longo, normal in _faces_da_parede(parede):
            pontos = _amostrar_retangulo(
                canto, ao_longo, np.array([0.0, 0.0, 1.0]), comprimento, ALTURA, passo
            )
            s = (pontos - canto) @ ao_longo + parede.inicio
            fora = _mascara_vaos(parede.nome, s, pontos[:, 2], extras) | _mascara_oclusao(
                parede.nome, s
            )
            pontos = pontos[~fora]
            pontos = pontos + normal * rng.normal(0.0, ruido, size=(len(pontos), 1))
            partes.append(pontos)

    # Piso e teto, so dentro do perimetro e **nunca por baixo de parede**: o
    # sensor nao ve atraves de alvenaria. A excecao e o vao de porta, onde o
    # piso atravessa a espessura da parede - e justamente essa diferenca que
    # permite distinguir porta de trecho ocluido.
    for altura, sinal in ((0.0, 1.0), (ALTURA, -1.0)):
        laje = _amostrar_retangulo(
            np.array([0.0, 0.0, altura]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            LARGURA,
            PROFUNDIDADE,
            passo * 2,
        )
        sob_parede = _dentro_de_parede(laje[:, :2], vaos_passantes=(altura == 0.0))
        laje = laje[~sob_parede]
        laje[:, 2] += sinal * rng.normal(0.0, ruido, size=len(laje))
        partes.append(laje)

    # Superficies que o sensor realmente ve num vao: peitoril, verga e as
    # ombreiras laterais, todas atravessando a espessura da parede.
    partes.extend(_superficies_dos_vaos(passo))
    if porta_fechada:
        partes.extend(_folha_de_porta_fechada(passo))

    # Moveis: caixas sem a face inferior, so as faces vistas pelo sensor.
    for x0, y0, x1, y1, altura in MOVEIS:
        tampo = _amostrar_retangulo(
            np.array([x0, y0, altura]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            x1 - x0,
            y1 - y0,
            passo * 1.5,
        )
        partes.append(tampo)
        for canto, u, du in (
            (np.array([x0, y0, 0.0]), np.array([1.0, 0.0, 0.0]), x1 - x0),
            (np.array([x0, y1, 0.0]), np.array([1.0, 0.0, 0.0]), x1 - x0),
            (np.array([x0, y0, 0.0]), np.array([0.0, 1.0, 0.0]), y1 - y0),
            (np.array([x1, y0, 0.0]), np.array([0.0, 1.0, 0.0]), y1 - y0),
        ):
            partes.append(
                _amostrar_retangulo(canto, u, np.array([0.0, 0.0, 1.0]), du, altura, passo * 1.5)
            )

    nuvem = np.concatenate(partes)

    # Pontos do lado de fora, para que as janelas do perimetro tenham "fundo":
    # o detector de vaos procura pontos atras do plano da parede.
    externo = rng.uniform(
        low=[-6.0, -6.0, 0.0], high=[LARGURA + 6.0, PROFUNDIDADE + 6.0, ALTURA], size=(60_000, 3)
    )
    fora = (
        (externo[:, 0] < -0.5)
        | (externo[:, 0] > LARGURA + 0.5)
        | (externo[:, 1] < -0.5)
        | (externo[:, 1] > PROFUNDIDADE + 0.5)
    )
    nuvem = np.concatenate([nuvem, externo[fora]])

    # Extrinseca "xyz": o desaprumo entra antes e o azimute gira em torno da
    # vertical do mundo, que e a ordem fisica de um levantamento inclinado.
    rotacao = Rotation.from_euler(
        "xyz", [desaprumo_graus[0], desaprumo_graus[1], azimute_graus], degrees=True
    ).as_matrix()
    return nuvem @ rotacao.T + np.asarray(translacao)


def paredes_esperadas() -> list[ParedeVerdade]:
    """Gabarito das paredes, para as asercoes do teste."""
    return list(PAREDES)


def vaos_esperados() -> list[Vao]:
    """Gabarito dos vaos, para as asercoes do teste."""
    return list(VAOS)


def salvar_las(caminho: str, nuvem: np.ndarray | None = None) -> str:
    """Grava a nuvem sintetica em LAS 1.4, para testar a leitura em blocos."""
    import laspy

    if nuvem is None:
        nuvem = gerar()
    cabecalho = laspy.LasHeader(point_format=6, version="1.4")
    cabecalho.scales = np.array([0.001, 0.001, 0.001])
    cabecalho.offsets = nuvem.min(axis=0)
    dados = laspy.LasData(cabecalho)
    dados.x, dados.y, dados.z = nuvem[:, 0], nuvem[:, 1], nuvem[:, 2]
    dados.write(caminho)
    return caminho


if __name__ == "__main__":
    pontos = gerar()
    print(f"{len(pontos)} pontos; limites {pontos.min(axis=0)} a {pontos.max(axis=0)}")
