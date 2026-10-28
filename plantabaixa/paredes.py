"""Etapas 2 e 3: mascara de parede e vetorizacao das faces.

A ideia central esta em `mapa_ocupacao`. Cortar a nuvem numa faixa de altura
fixa - digamos de 1,0 m a 2,0 m - e o caminho obvio e tambem o que falha em
ambiente mobiliado: um armario de 2 m aparece no corte exatamente como uma
parede apareceria. O que separa um do outro nao e a altura em que estao, e sim
*quanto do vao piso-teto cada coluna vertical ocupa*: a parede atravessa o vao
inteiro e encosta no teto, o armario para antes.

Por isso cada celula da grade XY recebe duas medidas:

- `score`: fracao dos bins de altura que a coluna ocupa (persistencia vertical);
- `teto`:  se a coluna tem pontos na faixa colada no teto.

Parede e a celula que satisfaz as duas. A parede atras do armario fica de fora -
ela esta ocluida -, vira lacuna no segmento e e resolvida na etapa 4, que sabe
distinguir lacuna por oclusao de lacuna por porta.

Toda a geometria desta etapa roda num *referencial do predio*, girado de `theta`
em relacao ao mundo, no qual as paredes ficam paralelas aos eixos. Isso reduz
deteccao de retas, medida de espessura e recorte de vaos a operacoes 1D.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from plantabaixa.erros import PlantaError

# Bits disponiveis na chave int64 que acumula os bins de altura ocupados.
MAX_BINS_ALTURA = 63


@dataclass
class Grade:
    """Grade de ocupacao 2D no referencial do predio.

    `score` e `teto` tem forma (ny, nx); `origem` e o canto de indice (0, 0).
    """

    score: np.ndarray
    teto: np.ndarray
    livre: np.ndarray
    res: float
    origem: np.ndarray
    theta: float
    bins_altura: int

    @property
    def forma(self) -> tuple[int, int]:
        return self.score.shape

    def mascara(self, persistencia: float) -> np.ndarray:
        """Celulas de parede: persistencia suficiente e contato com o teto."""
        return (self.score >= persistencia) & self.teto

    def celulas_xy(self, mascara: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Centros (x', y') das celulas marcadas e seus pesos (`score`)."""
        iy, ix = np.nonzero(mascara)
        xy = np.stack(
            [self.origem[0] + (ix + 0.5) * self.res, self.origem[1] + (iy + 0.5) * self.res],
            axis=1,
        )
        return xy, self.score[iy, ix]


@dataclass
class Lacuna:
    """Trecho de uma face sem pontos, ainda sem classificacao."""

    inicio: float
    fim: float

    @property
    def largura(self) -> float:
        return self.fim - self.inicio


@dataclass
class Face:
    """Uma face de parede: reta no referencial do predio, com extensao e lacunas.

    `normal . p = offset` para todo ponto da reta; `direcao` e perpendicular a
    `normal`. O sentido de `normal` so e fixado no emparelhamento, quando se sabe
    qual lado e o corpo da parede.
    """

    normal: np.ndarray
    direcao: np.ndarray
    offset: float
    s_min: float
    s_max: float
    lacunas: list[Lacuna] = field(default_factory=list)
    celulas: int = 0
    residuo: float = 0.0
    eixo: int = -1

    @property
    def comprimento(self) -> float:
        return self.s_max - self.s_min

    @property
    def comprimento_util(self) -> float:
        """Extensao com pontos, descontadas as lacunas."""
        return self.comprimento - sum(l.largura for l in self.lacunas)

    def ponto(self, s: float) -> np.ndarray:
        """Ponto da reta na abscissa `s`, no referencial do predio."""
        return self.normal * self.offset + self.direcao * s


@dataclass
class Parede:
    """Parede da planta: uma ou duas faces paralelas e sua espessura."""

    nome: str
    faces: list[Face]
    espessura: float
    face_unica: bool
    eixo_central: float
    normal: np.ndarray
    direcao: np.ndarray
    s_min: float
    s_max: float
    lacunas: list[Lacuna] = field(default_factory=list)

    @property
    def comprimento(self) -> float:
        return self.s_max - self.s_min

    def contorno(self) -> np.ndarray:
        """Quatro cantos do corpo da parede, no referencial do predio."""
        meia = self.espessura / 2
        base = self.normal * self.eixo_central
        return np.array(
            [
                base + self.normal * meia + self.direcao * self.s_min,
                base + self.normal * meia + self.direcao * self.s_max,
                base - self.normal * meia + self.direcao * self.s_max,
                base - self.normal * meia + self.direcao * self.s_min,
            ]
        )


def rotacao_2d(theta: float) -> np.ndarray:
    """Matriz 2x2 de rotacao de `theta` radianos."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def para_predio(xyz: np.ndarray, theta: float) -> np.ndarray:
    """Leva a nuvem do referencial local para o do predio, girando so em XY.

    Toda a geometria produzida por este modulo - offset de face, abscissa `s`,
    eixo de parede - vive no referencial do predio. Qualquer etapa seguinte que
    volte a tocar nos pontos precisa deste giro antes, sob pena de comparar
    coordenadas de referenciais diferentes.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    girada = xyz.copy()
    girada[:, :2] = xyz[:, :2] @ rotacao_2d(-theta).T
    return girada


def estimar_direcao(
    xy: np.ndarray, pesos: np.ndarray | None = None, res: float = 0.02, amostra: int = 40_000
) -> float:
    """Angulo `theta` que alinha as paredes dominantes aos eixos, em radianos.

    Procura o angulo que deixa os histogramas das projecoes em x e y mais
    concentrados: quando as paredes ficam paralelas aos eixos, cada parede
    colapsa num punhado de bins e a soma dos quadrados do histograma dispara.
    Busca grossa de 0,5 grau seguida de refino de 0,02 grau; a resposta e unica
    modulo 90 graus, entao a varredura cobre apenas [0, 90).
    """
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) < 10:
        raise PlantaError("Celulas de parede insuficientes para estimar a direcao dominante.")
    if pesos is None:
        pesos = np.ones(len(xy))
    if len(xy) > amostra:
        passo = len(xy) // amostra
        xy, pesos = xy[::passo], pesos[::passo]

    centro = xy.mean(axis=0)
    centrado = xy - centro
    raio = float(np.abs(centrado).max()) + res

    def nitidez(theta: float) -> float:
        girado = centrado @ rotacao_2d(-theta).T
        total = 0.0
        for coluna in (0, 1):
            hist, _ = np.histogram(
                girado[:, coluna], bins=int(2 * raio / res) + 1, range=(-raio, raio), weights=pesos
            )
            total += float(np.square(hist).sum())
        return total

    grossa = np.deg2rad(np.arange(0.0, 90.0, 0.5))
    melhor = max(grossa, key=nitidez)
    fina = melhor + np.deg2rad(np.arange(-0.5, 0.5001, 0.02))
    melhor = max(fina, key=nitidez)
    return float(melhor % (np.pi / 2))


def mapa_ocupacao(
    xyz: np.ndarray,
    z_teto: float,
    res: float = 0.02,
    theta: float = 0.0,
    bin_altura: float = 0.10,
    margem: float = 0.20,
    faixa_teto: float = 0.40,
) -> Grade:
    """Monta a grade de persistencia vertical no referencial do predio.

    `margem` descarta a faixa colada no piso e no teto, onde rodape, laje e
    luminaria poluem a contagem. `faixa_teto` e a espessura da faixa alta usada
    para decidir se a coluna encosta no teto.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if res <= 0:
        raise PlantaError(f"Resolucao de grade invalida: {res}.")
    z_inferior, z_superior = margem, z_teto - margem
    vao = z_superior - z_inferior
    if vao <= 3 * bin_altura:
        raise PlantaError(
            f"Vao util de apenas {vao:.2f} m entre piso e teto; "
            "reduza --margem ou confira o nivelamento da nuvem."
        )
    # A chave de bits cabe em int64, entao o bin cresce em ambiente muito alto.
    bin_altura = max(bin_altura, vao / MAX_BINS_ALTURA)
    n_bins = int(np.floor(vao / bin_altura))

    xy = xyz[:, :2] @ rotacao_2d(-theta).T
    origem = xy.min(axis=0) - res
    indices = np.floor((xy - origem) / res).astype(np.int64)
    nx = int(indices[:, 0].max()) + 2
    ny = int(indices[:, 1].max()) + 2
    plano = indices[:, 1] * nx + indices[:, 0]

    z = xyz[:, 2]
    na_faixa = (z >= z_inferior) & (z < z_superior)
    bins = np.clip(((z[na_faixa] - z_inferior) / bin_altura).astype(np.int64), 0, n_bins - 1)
    acumulador = np.zeros(ny * nx, dtype=np.int64)
    np.bitwise_or.at(acumulador, plano[na_faixa], np.int64(1) << bins)
    score = (np.bitwise_count(acumulador) / n_bins).astype(np.float32).reshape(ny, nx)

    alto = z >= max(z_teto - faixa_teto, z_inferior)
    teto = np.zeros(ny * nx, dtype=bool)
    teto[plano[alto]] = True

    # Espaco livre: celula com piso visivel. Serve para saber de que lado de uma
    # parede de face unica esta o ambiente, sem precisar segmentar salas.
    piso = np.abs(z) <= 0.10
    livre = np.zeros(ny * nx, dtype=bool)
    livre[plano[piso]] = True

    return Grade(
        score=score,
        teto=teto.reshape(ny, nx),
        livre=livre.reshape(ny, nx),
        res=res,
        origem=origem,
        theta=theta,
        bins_altura=n_bins,
    )


def orientacao_local(
    mascara: np.ndarray, res: float, sigma_gradiente: float = 0.04, sigma_janela: float = 0.12
) -> tuple[np.ndarray, np.ndarray]:
    """Angulo da normal e coerencia de cada celula, pelo tensor de estrutura.

    Sem esta etapa, o histograma de deslocamento de um eixo recebe como fundo as
    celulas de todas as paredes perpendiculares a ele: uma parede de 16 m espalha
    800 celulas pelo eixo transversal, altura de fundo suficiente para fabricar
    picos onde nao ha parede alguma. Classificando cada celula pela orientacao
    local da estrutura a que pertence, cada eixo recebe apenas as suas.

    O angulo e devolvido modulo pi (normal e sua oposta sao equivalentes) e a
    coerencia vale 1 numa reta perfeita e 0 onde nao ha direcao dominante -
    cruzamentos de parede e pontos de bagunca, que ficam de fora da vetorizacao.
    """
    from scipy.ndimage import gaussian_filter

    campo = gaussian_filter(mascara.astype(np.float32), sigma_gradiente / res)
    gy, gx = np.gradient(campo)
    janela = sigma_janela / res
    jxx = gaussian_filter(gx * gx, janela)
    jyy = gaussian_filter(gy * gy, janela)
    jxy = gaussian_filter(gx * gy, janela)
    angulo = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    traco = jxx + jyy
    discriminante = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy**2)
    coerencia = np.divide(discriminante, traco, out=np.zeros_like(traco), where=traco > 1e-12)
    return angulo, coerencia


def classificar_celulas(
    angulo: np.ndarray,
    coerencia: np.ndarray,
    mascara: np.ndarray,
    tol_angulo_graus: float = 12.0,
    coerencia_minima: float = 0.25,
) -> np.ndarray:
    """Rotula cada celula de parede: 0 e 1 para os eixos, 2 para obliqua, -1 descartada."""
    rotulo = np.full(mascara.shape, -1, dtype=np.int8)
    confiavel = mascara & (coerencia >= coerencia_minima)
    limite = np.cos(np.deg2rad(tol_angulo_graus))
    cos_a, sen_a = np.abs(np.cos(angulo)), np.abs(np.sin(angulo))
    rotulo[confiavel & (cos_a >= limite)] = 0
    rotulo[confiavel & (sen_a >= limite)] = 1
    rotulo[confiavel & (cos_a < limite) & (sen_a < limite)] = 2
    return rotulo


# Medido na nuvem de gabarito: face de parede real cobre de 36% a 87% da sua
# extensao - e nao quase 100%, porque o ruido faz a face alternar entre duas
# linhas vizinhas da grade e cada linha pega so parte do comprimento. Reta
# espuria do RANSAC obliquo fica em torno de 9%. O limiar fica no vao entre as
# duas populacoes, com folga para os dois lados.
DENSIDADE_MINIMA = 0.20


def _segmentar(
    s: np.ndarray,
    res: float,
    lacuna_minima: float,
    comprimento_minimo: float,
    densidade_minima: float = DENSIDADE_MINIMA,
) -> tuple[float, float, list[Lacuna], float] | None:
    """Extensao, lacunas e comprimento util de um conjunto de abscissas.

    Lacunas menores que `lacuna_minima` sao costuradas: vem de oclusao pontual,
    de rodape ou de ruido, e nao de abertura.

    `densidade_minima` descarta a reta que apenas *passa* por celulas esparsas em
    vez de percorre-las. Sem esse filtro o RANSAC obliquo encontra alinhamentos
    fortuitos em celulas de canto e bagunca, e inventa parede onde ha ruido. A
    fracao e medida contra a extensao *total*, nao contra o comprimento util -
    medir contra o util seria circular, ja que ele e definido descontando
    exatamente os vazios que se quer penalizar.
    """
    if not len(s):
        return None
    s_min, s_max = float(s.min()), float(s.max())
    n = max(1, int(np.ceil((s_max - s_min) / res)) + 1)
    ocupado = np.zeros(n, dtype=bool)
    ocupado[np.clip(((s - s_min) / res).astype(np.int64), 0, n - 1)] = True

    # Bordas das corridas de celulas vazias.
    borda = np.diff(np.concatenate([[True], ocupado, [True]]).astype(np.int8))
    inicios = np.flatnonzero(borda == -1)
    fins = np.flatnonzero(borda == 1)
    lacunas = [
        Lacuna(s_min + i * res, s_min + f * res)
        for i, f in zip(inicios, fins)
        if (f - i) * res >= lacuna_minima
    ]
    util = (s_max - s_min) - sum(l.largura for l in lacunas)
    if util < comprimento_minimo:
        return None
    if float(ocupado.sum()) * res < densidade_minima * (s_max - s_min):
        return None
    return s_min, s_max, lacunas, util


def _faces_do_eixo(
    xy: np.ndarray,
    pesos: np.ndarray,
    eixo: int,
    res: float,
    tol_face: float,
    lacuna: float,
    min_parede: float,
) -> list[Face]:
    """Faces perpendiculares a `eixo`, por histograma de deslocamento.

    Recebe apenas as celulas que `classificar_celulas` atribuiu a este eixo, o
    que deixa o histograma praticamente sem fundo: cada pico e uma face.
    """
    from scipy.signal import find_peaks

    normal = np.array([1.0, 0.0]) if eixo == 0 else np.array([0.0, 1.0])
    direcao = np.array([0.0, 1.0]) if eixo == 0 else np.array([1.0, 0.0])
    deslocamento = xy @ normal
    abscissa = xy @ direcao

    from scipy.ndimage import gaussian_filter1d

    # Histograma fino suavizado por nucleo gaussiano, em vez de contagem crua.
    # Com bins menores que a celula da grade, o histograma sai em forma de pente
    # - rajadas separadas por bins vazios - e o detector de picos passa a
    # depender de onde a borda do bin caiu, perdendo faces inteiras. A largura do
    # nucleo acompanha o que de fato espalha a face: o ruido do levantamento
    # (~1 cm) somado a quantizacao da grade.
    passo = 0.005
    sigma = max(0.015, 0.7 * res)
    low = float(deslocamento.min()) - 4 * sigma
    high = float(deslocamento.max()) + 4 * sigma
    n_bins = max(8, int(np.ceil((high - low) / passo)))
    hist, bordas = np.histogram(deslocamento, bins=n_bins, range=(low, high), weights=pesos)
    centros = 0.5 * (bordas[:-1] + bordas[1:])
    # O fator devolve a densidade a escala de "celulas sob o pico", para que o
    # limiar continue sendo lido em numero de celulas.
    suave = gaussian_filter1d(hist.astype(float), sigma / passo) * (
        sigma * np.sqrt(2 * np.pi) / passo
    )

    # Uma parede de comprimento minimo rende min_parede/res celulas; exige-se
    # 40% disso para o pico valer como face, e o resto e filtrado adiante por
    # comprimento e densidade.
    altura_minima = max(3.0, 0.4 * min_parede / res)
    picos, _ = find_peaks(
        suave,
        height=altura_minima,
        prominence=altura_minima / 2,
        distance=max(1, int(0.05 / passo)),
    )

    faces: list[Face] = []
    for pico in picos:
        offset = float(centros[pico])
        perto = np.abs(deslocamento - offset) <= tol_face
        if not perto.any():
            continue
        # Reajusta o offset pela mediana dos pontos atribuidos.
        offset = float(np.median(deslocamento[perto]))
        perto = np.abs(deslocamento - offset) <= tol_face
        segmentos = _segmentar(abscissa[perto], res, lacuna, min_parede)
        if segmentos is None:
            continue
        s_min, s_max, lacunas, _ = segmentos
        residuo = float(np.sqrt(np.mean((deslocamento[perto] - offset) ** 2)))
        faces.append(
            Face(
                normal=normal.copy(),
                direcao=direcao.copy(),
                offset=offset,
                s_min=s_min,
                s_max=s_max,
                lacunas=lacunas,
                celulas=int(perto.sum()),
                residuo=residuo,
                eixo=eixo,
            )
        )
    return _fundir_faces(faces, tol_face)


def _fundir_faces(faces: list[Face], tol_face: float) -> list[Face]:
    """Funde faces do mesmo eixo que acabaram no mesmo lugar.

    O reajuste do `offset` pela mediana dos pontos atribuidos pode aproximar dois
    picos vizinhos ate que descrevam a mesma face; sem esta fusao, a mesma parede
    viraria duas, e o emparelhamento mediria uma espessura inexistente.
    """
    if not faces:
        return []
    ordenadas = sorted(faces, key=lambda f: f.offset)
    fundidas = [ordenadas[0]]
    for face in ordenadas[1:]:
        anterior = fundidas[-1]
        if abs(face.offset - anterior.offset) <= tol_face and _sobreposicao(anterior, face) > 0.2:
            # Mantem a face com mais celulas e estende a extensao coberta.
            vencedora, perdedora = (
                (anterior, face) if anterior.celulas >= face.celulas else (face, anterior)
            )
            vencedora.s_min = min(anterior.s_min, face.s_min)
            vencedora.s_max = max(anterior.s_max, face.s_max)
            vencedora.celulas = anterior.celulas + perdedora.celulas
            fundidas[-1] = vencedora
        else:
            fundidas.append(face)
    return fundidas


def _faces_obliquas(
    xy: np.ndarray,
    res: float,
    tol_face: float,
    lacuna: float,
    min_parede: float,
    max_faces: int = 12,
    semente: int = 55,
) -> list[Face]:
    """RANSAC 2D nas celulas que nenhum eixo explicou, para paredes fora de esquadro."""
    rng = np.random.default_rng(semente)
    restante = xy.copy()
    faces: list[Face] = []
    minimo_celulas = max(8, int(min_parede / res / 2))

    while len(restante) >= minimo_celulas and len(faces) < max_faces:
        melhor: tuple[int, np.ndarray, float] | None = None
        for _ in range(300):
            a, b = restante[rng.choice(len(restante), size=2, replace=False)]
            direcao = b - a
            norma = np.linalg.norm(direcao)
            if norma < min_parede:
                continue
            direcao = direcao / norma
            normal = np.array([-direcao[1], direcao[0]])
            offset = float(normal @ a)
            inliers = np.abs(restante @ normal - offset) <= tol_face
            n = int(inliers.sum())
            if melhor is None or n > melhor[0]:
                melhor = (n, normal, offset)
        if melhor is None or melhor[0] < minimo_celulas:
            break
        _, normal, offset = melhor
        direcao = np.array([normal[1], -normal[0]])
        inliers = np.abs(restante @ normal - offset) <= tol_face
        segmentos = _segmentar(restante[inliers] @ direcao, res, lacuna, min_parede)
        if segmentos is not None:
            s_min, s_max, lacunas, _ = segmentos
            faces.append(
                Face(
                    normal=normal,
                    direcao=direcao,
                    offset=offset,
                    s_min=s_min,
                    s_max=s_max,
                    lacunas=lacunas,
                    celulas=int(inliers.sum()),
                    residuo=float(np.sqrt(np.mean((restante[inliers] @ normal - offset) ** 2))),
                    eixo=-1,
                )
            )
        restante = restante[~inliers]
    return faces


def _lado_livre(grade: Grade, face: Face, distancia: float) -> int:
    """+1 ou -1: de que lado da face ha mais piso visivel.

    Amostra pontos ao longo da face, deslocados de `distancia` para cada lado, e
    conta quantos caem em celula com piso. Usado para decidir para onde cresce o
    corpo de uma parede vista de um lado so.
    """
    amostras = np.linspace(face.s_min, face.s_max, max(8, int(face.comprimento / 0.25)))
    pontos = face.normal * face.offset + np.outer(amostras, face.direcao)
    contagem = []
    for sinal in (1.0, -1.0):
        deslocados = pontos + sinal * face.normal * distancia
        ij = np.floor((deslocados - grade.origem) / grade.res).astype(np.int64)
        dentro = (
            (ij[:, 0] >= 0)
            & (ij[:, 1] >= 0)
            & (ij[:, 0] < grade.forma[1])
            & (ij[:, 1] < grade.forma[0])
        )
        contagem.append(int(grade.livre[ij[dentro, 1], ij[dentro, 0]].sum()))
    return 1 if contagem[0] >= contagem[1] else -1


def _sobreposicao(a: Face, b: Face) -> float:
    """Fracao da face mais curta coberta pela sobreposicao das duas extensoes."""
    inicio = max(a.s_min, b.s_min)
    fim = min(a.s_max, b.s_max)
    menor = min(a.comprimento, b.comprimento)
    if menor <= 0:
        return 0.0
    return max(0.0, fim - inicio) / menor


def emparelhar(
    faces: list[Face],
    grade: Grade,
    espessura_min: float = 0.08,
    espessura_max: float = 0.45,
    espessura_padrao: float = 0.15,
    sobreposicao_minima: float = 0.35,
) -> list[Parede]:
    """Agrupa faces paralelas e proximas na mesma parede, medindo a espessura.

    Duas faces viradas uma para a outra, separadas por 8 a 35 cm e com extensoes
    que se sobrepoem, sao os dois lados de uma parede de alvenaria. Face sem par
    recebe `espessura_padrao` e e marcada: o corpo da parede cresce para o lado
    sem piso visivel, que e o lado nao levantado.
    """
    restantes = sorted(faces, key=lambda f: (-f.comprimento_util, f.offset))
    paredes: list[Parede] = []
    consumidas: set[int] = set()

    for i, face in enumerate(restantes):
        if i in consumidas:
            continue
        par: tuple[int, Face] | None = None
        for j in range(i + 1, len(restantes)):
            if j in consumidas:
                continue
            outra = restantes[j]
            # Paralelas: normais com mesma direcao a menos de sinal.
            if abs(abs(float(face.normal @ outra.normal)) - 1.0) > 1e-3:
                continue
            sinal = 1.0 if float(face.normal @ outra.normal) > 0 else -1.0
            separacao = abs(face.offset - sinal * outra.offset)
            if not espessura_min <= separacao <= espessura_max:
                continue
            if _sobreposicao(face, outra) < sobreposicao_minima:
                continue
            if par is None or separacao < abs(face.offset - par[1].offset):
                par = (j, outra)
        if par is not None:
            j, outra = par
            consumidas.add(j)
            sinal = 1.0 if float(face.normal @ outra.normal) > 0 else -1.0
            offset_outra = sinal * outra.offset
            espessura = abs(face.offset - offset_outra)
            eixo_central = 0.5 * (face.offset + offset_outra)
            s_min = min(face.s_min, outra.s_min)
            s_max = max(face.s_max, outra.s_max)
            paredes.append(
                Parede(
                    nome=f"P{len(paredes) + 1:02d}",
                    faces=[face, outra],
                    espessura=espessura,
                    face_unica=False,
                    eixo_central=eixo_central,
                    normal=face.normal.copy(),
                    direcao=face.direcao.copy(),
                    s_min=s_min,
                    s_max=s_max,
                    lacunas=_lacunas_comuns(face, outra),
                )
            )
        else:
            lado = _lado_livre(grade, face, espessura_padrao + 2 * grade.res)
            eixo_central = face.offset - lado * espessura_padrao / 2
            paredes.append(
                Parede(
                    nome=f"P{len(paredes) + 1:02d}",
                    faces=[face],
                    espessura=espessura_padrao,
                    face_unica=True,
                    eixo_central=eixo_central,
                    normal=face.normal.copy(),
                    direcao=face.direcao.copy(),
                    s_min=face.s_min,
                    s_max=face.s_max,
                    lacunas=list(face.lacunas),
                )
            )
        consumidas.add(i)
    return paredes


def _lacunas_comuns(a: Face, b: Face) -> list[Lacuna]:
    """Interseccao das lacunas das duas faces.

    So interessa a lacuna que atravessa a parede: um trecho vazio em uma face e
    cheio na outra e oclusao daquele lado, nao abertura.
    """
    comuns: list[Lacuna] = []
    for la in a.lacunas:
        for lb in b.lacunas:
            inicio, fim = max(la.inicio, lb.inicio), min(la.fim, lb.fim)
            if fim - inicio > 0:
                comuns.append(Lacuna(inicio, fim))
    return comuns


def ajustar_cantos(paredes: list[Parede], tolerancia: float = 0.60) -> list[str]:
    """Estende cada ponta de parede ate a parede perpendicular que a encontra.

    As pontas chegam aparadas da vetorizacao: numa juncao em L ou em T a
    orientacao local deixa de ter direcao dominante e aquelas celulas sao
    descartadas, o que abre uma falha de 10 a 20 cm em cada canto. A ponta e
    levada ate a *face distante* da parede encontrada, de modo que o canto fique
    macico - e nao apenas encostado.

    Devolve a lista de cantos ajustados, para o relatorio.
    """
    ajustes: list[str] = []
    for parede in paredes:
        for ponta in ("s_min", "s_max"):
            valor = getattr(parede, ponta)
            sinal = -1.0 if ponta == "s_min" else 1.0
            melhor: tuple[float, Parede, float] | None = None
            for outra in paredes:
                if outra is parede:
                    continue
                if abs(float(parede.normal @ outra.normal)) > 1e-3:
                    continue  # paralelas: nao formam canto
                # Interseccao dos eixos: as normais sao ortogonais e unitarias.
                ponto = parede.normal * parede.eixo_central + outra.normal * outra.eixo_central
                s_cruz = float(parede.direcao @ ponto)
                t_cruz = float(outra.direcao @ ponto)
                if abs(s_cruz - valor) > tolerancia:
                    continue
                if not (outra.s_min - tolerancia <= t_cruz <= outra.s_max + tolerancia):
                    continue
                if melhor is None or abs(s_cruz - valor) < abs(melhor[0] - valor):
                    melhor = (s_cruz, outra, t_cruz)
            if melhor is None:
                continue
            s_cruz, outra, _ = melhor
            novo = s_cruz + sinal * outra.espessura / 2
            # So estende; nunca encurta uma parede que ja passa do cruzamento.
            if sinal * (novo - valor) > 0:
                setattr(parede, ponta, novo)
                ajustes.append(
                    f"{parede.nome}.{ponta} estendida {abs(novo - valor) * 100:.0f} cm "
                    f"ate {outra.nome}"
                )
    return ajustes


@dataclass
class Extracao:
    """Resultado das etapas 2 e 3, com o material de diagnostico."""

    paredes: list[Parede]
    grade: Grade
    mascara: np.ndarray
    rotulos: np.ndarray
    faces: list[Face]
    theta: float
    xyz_predio: np.ndarray
    ajustes_canto: list[str] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)


def extrair_paredes(
    xyz: np.ndarray,
    z_teto: float,
    res: float = 0.02,
    persistencia: float = 0.50,
    lacuna: float = 0.30,
    min_parede: float = 0.40,
    tol_face: float = 0.04,
    tol_angulo: float = 12.0,
    espessura_max: float = 0.45,
    espessura_padrao: float = 0.15,
    manhattan: bool = True,
    fechamento: int = 1,
) -> Extracao:
    """Encadeia as etapas 2 e 3: da nuvem nivelada as paredes vetorizadas."""
    from scipy.ndimage import binary_closing

    grade_mundo = mapa_ocupacao(xyz, z_teto, res=res, theta=0.0)
    mascara = grade_mundo.mascara(persistencia)
    if not mascara.any():
        raise PlantaError(
            f"Nenhuma celula passou o critério de parede (persistência >= {persistencia}). "
            "Reduza --persistencia ou confira se a nuvem cobre do piso ao teto."
        )
    xy_mundo, pesos_mundo = grade_mundo.celulas_xy(mascara)
    theta = estimar_direcao(xy_mundo, pesos_mundo, res=res) if manhattan else 0.0

    # Refaz a grade no referencial do predio: com as paredes paralelas aos eixos,
    # a deteccao de retas e o recorte dos vaos ficam unidimensionais.
    grade = mapa_ocupacao(xyz, z_teto, res=res, theta=theta)
    mascara = grade.mascara(persistencia)
    if fechamento > 0:
        mascara = binary_closing(mascara, iterations=fechamento)

    angulo, coerencia = orientacao_local(mascara, res)
    rotulos = classificar_celulas(angulo, coerencia, mascara, tol_angulo_graus=tol_angulo)

    faces: list[Face] = []
    for eixo in (0, 1):
        if not manhattan:
            break
        xy, pesos = grade.celulas_xy(rotulos == eixo)
        if len(xy) < 8:
            continue
        faces.extend(_faces_do_eixo(xy, pesos, eixo, res, tol_face, lacuna, min_parede))

    obliquas = rotulos == 2 if manhattan else mascara
    if obliquas.sum() >= 8:
        xy, _ = grade.celulas_xy(obliquas)
        faces.extend(_faces_obliquas(xy, res, tol_face, lacuna, min_parede))

    if not faces:
        raise PlantaError("Nenhuma face de parede detectada; confira os diagnósticos da grade.")

    paredes = emparelhar(
        faces,
        grade,
        espessura_max=espessura_max,
        espessura_padrao=espessura_padrao,
    )
    ajustes = ajustar_cantos(paredes)

    # Uma parede so e medida com confianca se couberem umas tres celulas na sua
    # espessura; abaixo disso as duas faces encostam na grade e a espessura vira
    # artefato da resolucao, nao medida.
    avisos: list[str] = []
    finas = [p for p in paredes if not p.face_unica and p.espessura < 3 * res]
    if finas:
        avisos.append(
            f"{len(finas)} parede(s) com espessura abaixo de 3 celulas da grade "
            f"({', '.join(p.nome for p in finas)}); reduza --res para medi-las "
            f"com confianca (atual: {res * 100:.0f} cm)."
        )

    return Extracao(
        paredes=paredes,
        grade=grade,
        mascara=mascara,
        rotulos=rotulos,
        faces=faces,
        theta=theta,
        xyz_predio=para_predio(xyz, theta),
        ajustes_canto=ajustes,
        avisos=avisos,
    )
