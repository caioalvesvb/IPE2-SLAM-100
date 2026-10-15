"""Etapa 1: carga, voxelizacao e nivelamento da nuvem registrada.

A nuvem exportada pelo SLAM GO POST de um levantamento de ~13 min traz entre 30
e 80 milhoes de pontos. Ler tudo de uma vez estoura a memoria, entao a leitura e
feita em blocos e cada bloco e reduzido a voxels imediatamente: o que interessa
para uma planta baixa e a ocupacao do espaco, nao a densidade do levantamento.

O nivelamento nao e opcional. Toda a extracao de paredes assume que `z` e a
vertical verdadeira e que o piso esta em `z = 0`; um desaprumo de 1 grau desloca
o teto em 50 cm numa sala de 30 m e desmonta o corte por altura.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from plantabaixa.erros import PlantaError

EXTENSOES = (".las", ".laz")
# Blocos de 4 M pontos: ~100 MB por eixo em float64, antes da voxelizacao.
PONTOS_POR_BLOCO = 4_000_000
# Chaves de voxel empacotadas em int64: 21 bits por eixo = 2.097.152 voxels,
# ou 41 km de extensao por eixo com voxel de 2 cm. Folga de sobra.
BITS_POR_EIXO = 21
LIMITE_INDICE = 1 << BITS_POR_EIXO


@dataclass
class Nuvem:
    """Nuvem voxelizada e nivelada, em metros, com o piso em `z = 0`.

    `pesos` guarda quantos pontos originais cairam em cada voxel; serve para
    ponderar ajustes e para o diagnostico de densidade, nao para geometria.
    """

    xyz: np.ndarray
    pesos: np.ndarray
    voxel: float
    z_teto: float
    rotacao: np.ndarray
    desnivel_graus: float
    origem_mundo: np.ndarray = field(default_factory=lambda: np.zeros(3))
    origem: Path | None = None
    pontos_lidos: int = 0
    avisos: list[str] = field(default_factory=list)

    def para_mundo(self, xyz: np.ndarray) -> np.ndarray:
        """Devolve coordenadas locais ao referencial original da nuvem.

        Inverte a translacao local e o nivelamento, na ordem contraria. Serve
        para gravar o DXF georreferenciado quando o cliente pede as coordenadas
        de projeto em vez da planta em coordenadas locais.
        """
        xyz = np.asarray(xyz, dtype=np.float64)
        return (xyz + self.origem_mundo) @ self.rotacao

    @property
    def altura_livre(self) -> float:
        """Distancia piso-teto, em metros."""
        return float(self.z_teto)

    def limites(self) -> tuple[np.ndarray, np.ndarray]:
        """Canto minimo e maximo da nuvem, em metros."""
        return self.xyz.min(axis=0), self.xyz.max(axis=0)


def descobrir_nuvem(caminho: str | Path) -> Path:
    """Resolve um arquivo `.las`/`.laz`, aceitando pasta de projeto.

    Numa pasta, prefere nomes que o SLAM GO POST costuma gerar (`result`,
    `filtered`) e, no empate, o maior arquivo - o export completo, nao um
    recorte de amostra.
    """
    caminho = Path(caminho)
    if caminho.is_file():
        if caminho.suffix.lower() not in EXTENSOES:
            raise PlantaError(f"Extensao nao suportada: {caminho.name}; use .las ou .laz.")
        return caminho
    if not caminho.is_dir():
        raise PlantaError(f"Caminho inexistente: {caminho}")

    candidatos = sorted(
        p for p in caminho.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSOES
    )
    if not candidatos:
        raise PlantaError(
            f"Nenhum .las ou .laz em {caminho}. "
            "Exporte a nuvem registrada no SLAM GO POST antes de gerar a planta."
        )

    def prioridade(p: Path) -> tuple[int, int]:
        nome = p.name.lower()
        preferido = any(marca in nome for marca in ("result", "filtered", "cloud"))
        return (0 if preferido else 1, -p.stat().st_size)

    return min(candidatos, key=prioridade)


def _empacotar(indices: np.ndarray) -> np.ndarray:
    """Reduz indices inteiros (N,3) a uma chave int64 por voxel."""
    if indices.min() < 0 or indices.max() >= LIMITE_INDICE:
        raise PlantaError(
            "Extensao da nuvem excede o empacotamento de voxels; "
            "aumente --voxel ou recorte a area com --recorte."
        )
    ix, iy, iz = indices.T.astype(np.int64)
    return (ix << (2 * BITS_POR_EIXO)) | (iy << BITS_POR_EIXO) | iz


def _desempacotar(chaves: np.ndarray) -> np.ndarray:
    """Recupera os indices (N,3) a partir das chaves int64."""
    mascara = LIMITE_INDICE - 1
    ix = chaves >> (2 * BITS_POR_EIXO)
    iy = (chaves >> BITS_POR_EIXO) & mascara
    iz = chaves & mascara
    return np.stack([ix, iy, iz], axis=1)


def _comprimir(chaves: list[np.ndarray], contagens: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Funde listas de chaves/contagens parciais somando os voxels repetidos."""
    todas = np.concatenate(chaves)
    pesos = np.concatenate(contagens)
    unicas, inverso = np.unique(todas, return_inverse=True)
    return unicas, np.bincount(inverso, weights=pesos).astype(np.int64)


def voxelizar(
    xyz: np.ndarray, voxel: float, origem: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Reduz pontos a centros de voxel unicos, devolvendo tambem a contagem.

    Funcao independente da leitura em blocos para poder ser testada e usada em
    nuvens que ja estejam em memoria (sinteticas, por exemplo).
    """
    if voxel <= 0:
        raise PlantaError(f"Tamanho de voxel invalido: {voxel}.")
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise PlantaError("A nuvem deve ter forma (N, 3).")
    if not len(xyz):
        raise PlantaError("Nuvem vazia.")
    if origem is None:
        origem = xyz.min(axis=0)
    indices = np.floor((xyz - origem) / voxel).astype(np.int64)
    chaves, contagens = np.unique(_empacotar(indices), return_counts=True)
    centros = (_desempacotar(chaves) + 0.5) * voxel + origem
    return centros, contagens.astype(np.int64)


def _ler_blocos(arquivo: Path, voxel: float, recorte: tuple[float, float, float, float] | None):
    """Le o arquivo em blocos e devolve (centros, pesos, pontos_lidos)."""
    import laspy

    try:
        leitor = laspy.open(arquivo)
    except Exception as erro:  # laspy levanta excecoes variadas por formato
        raise PlantaError(f"Falha ao abrir {arquivo.name}: {erro}") from erro

    with leitor as fonte:
        cabecalho = fonte.header
        if cabecalho.point_count == 0:
            raise PlantaError(f"{arquivo.name} nao tem pontos.")
        # A origem vem do cabecalho para que a grade de voxels seja a mesma em
        # todos os blocos, condicao para a deduplicacao funcionar.
        origem = np.asarray(cabecalho.mins, dtype=np.float64) - voxel
        chaves: list[np.ndarray] = []
        contagens: list[np.ndarray] = []
        lidos = 0
        for bloco in fonte.chunk_iterator(PONTOS_POR_BLOCO):
            xyz = np.stack(
                [
                    np.asarray(bloco.x, dtype=np.float64),
                    np.asarray(bloco.y, dtype=np.float64),
                    np.asarray(bloco.z, dtype=np.float64),
                ],
                axis=1,
            )
            lidos += len(xyz)
            if recorte is not None:
                x0, y0, x1, y1 = recorte
                dentro = (
                    (xyz[:, 0] >= x0)
                    & (xyz[:, 0] <= x1)
                    & (xyz[:, 1] >= y0)
                    & (xyz[:, 1] <= y1)
                )
                xyz = xyz[dentro]
            if not len(xyz):
                continue
            indices = np.floor((xyz - origem) / voxel).astype(np.int64)
            k, c = np.unique(_empacotar(indices), return_counts=True)
            chaves.append(k)
            contagens.append(c.astype(np.int64))
            # Comprime periodicamente para nao acumular blocos redundantes.
            if len(chaves) >= 8:
                k, c = _comprimir(chaves, contagens)
                chaves, contagens = [k], [c]

    if not chaves:
        raise PlantaError("Nenhum ponto sobrou apos o recorte; confira --recorte.")
    unicas, pesos = _comprimir(chaves, contagens)
    centros = (_desempacotar(unicas) + 0.5) * voxel + origem
    return centros, pesos, lidos


def ajustar_plano(
    pontos: np.ndarray,
    tolerancia: float = 0.05,
    iteracoes: int = 500,
    normal_minima_z: float = 0.90,
    semente: int = 20260827,
) -> tuple[np.ndarray, float, np.ndarray]:
    """RANSAC de plano quase horizontal, seguido de refino por minimos quadrados.

    Devolve `(normal, deslocamento, inliers)` com `normal . p = deslocamento` e
    `normal[2] > 0`. O vinculo `normal_minima_z` descarta amostras que caem em
    superficies verticais - sem ele, uma parede proxima rouba o ajuste do piso.
    """
    pontos = np.asarray(pontos, dtype=np.float64)
    if len(pontos) < 3:
        raise PlantaError("Pontos insuficientes para ajustar o plano do piso.")

    rng = np.random.default_rng(semente)
    melhor_normal = np.array([0.0, 0.0, 1.0])
    melhor_deslocamento = float(np.median(pontos[:, 2]))
    melhor_inliers = np.abs(pontos[:, 2] - melhor_deslocamento) <= tolerancia

    for _ in range(iteracoes):
        amostra = pontos[rng.choice(len(pontos), size=3, replace=False)]
        normal = np.cross(amostra[1] - amostra[0], amostra[2] - amostra[0])
        norma = np.linalg.norm(normal)
        if norma < 1e-9:
            continue
        normal = normal / norma
        if normal[2] < 0:
            normal = -normal
        if normal[2] < normal_minima_z:
            continue
        deslocamento = float(normal @ amostra[0])
        inliers = np.abs(pontos @ normal - deslocamento) <= tolerancia
        if inliers.sum() > melhor_inliers.sum():
            melhor_normal, melhor_deslocamento, melhor_inliers = normal, deslocamento, inliers

    # Refino: tres passes de minimos quadrados totais sobre os inliers, com a
    # tolerancia apertando, para tirar o vies da amostra de tres pontos.
    for tol in (tolerancia, tolerancia * 0.6, tolerancia * 0.4):
        selecao = pontos[melhor_inliers]
        if len(selecao) < 3:
            break
        centroide = selecao.mean(axis=0)
        _, _, vt = np.linalg.svd(selecao - centroide, full_matrices=False)
        normal = vt[-1]
        if normal[2] < 0:
            normal = -normal
        if normal[2] < normal_minima_z:
            break
        melhor_normal = normal
        melhor_deslocamento = float(normal @ centroide)
        melhor_inliers = np.abs(pontos @ melhor_normal - melhor_deslocamento) <= tol

    return melhor_normal, melhor_deslocamento, melhor_inliers


def _rotacao_para_vertical(normal: np.ndarray) -> np.ndarray:
    """Matriz que leva `normal` ao eixo +Z pelo caminho mais curto."""
    from scipy.spatial.transform import Rotation

    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    vertical = np.array([0.0, 0.0, 1.0])
    eixo = np.cross(normal, vertical)
    seno = float(np.linalg.norm(eixo))
    if seno < 1e-12:
        return np.eye(3)
    angulo = float(np.arctan2(seno, float(normal @ vertical)))
    return Rotation.from_rotvec(eixo / seno * angulo).as_matrix()


def _picos_altura(z: np.ndarray, pesos: np.ndarray, largura_bin: float = 0.05):
    """Histograma de altura ponderado, usado para achar piso e teto."""
    z_min, z_max = float(z.min()), float(z.max())
    n_bins = max(4, int(np.ceil((z_max - z_min) / largura_bin)))
    contagem, bordas = np.histogram(z, bins=n_bins, range=(z_min, z_max), weights=pesos)
    centros = 0.5 * (bordas[:-1] + bordas[1:])
    return centros, contagem


def detectar_teto(z: np.ndarray, pesos: np.ndarray, fracao_minima: float = 0.10) -> float:
    """Altura do teto: pico de altura mais alto com massa relevante.

    Procura acima de 1,8 m para nao confundir o teto com o tampo de armarios, e
    exige `fracao_minima` do pico dominante para ignorar luminarias e dutos. O
    valor devolvido e a mediana dos pontos ao redor do pico, nao o centro do bin.
    """
    centros, contagem = _picos_altura(z, pesos)
    acima = centros >= 1.80
    if not acima.any() or contagem[acima].max() <= 0:
        # Ambiente baixo ou nuvem so do piso: cai para o topo observado.
        return float(np.percentile(z, 99.5))
    limite = contagem.max() * fracao_minima
    candidatos = np.flatnonzero(acima & (contagem >= limite))
    if not len(candidatos):
        return float(np.percentile(z, 99.5))
    pico = float(centros[candidatos[-1]])
    perto = np.abs(z - pico) <= 0.08
    return float(np.median(z[perto])) if perto.any() else pico


def _detectar_niveis(z: np.ndarray, pesos: np.ndarray, z_teto: float) -> list[float]:
    """Alturas de piso candidatas, para avisar sobre escada ou outro pavimento."""
    centros, contagem = _picos_altura(z, pesos)
    if contagem.max() <= 0:
        return []
    limite = contagem.max() * 0.25
    # Um segundo pavimento aparece como massa forte acima do teto deste nivel.
    acima = np.flatnonzero((centros > z_teto + 0.5) & (contagem >= limite))
    return [float(centros[i]) for i in acima]


def nivelar(
    xyz: np.ndarray, pesos: np.ndarray, percentil_piso: float = 25.0
) -> tuple[np.ndarray, np.ndarray, float, float, list[str]]:
    """Gira a nuvem para o piso ficar horizontal e translada o piso para z = 0.

    O piso e procurado no quartil inferior de altura em vez do pico do
    histograma: com a nuvem desaprumada, o pico do piso fica esborrachado, mas o
    quartil inferior continua dominado por pontos de piso.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    avisos: list[str] = []
    corte = float(np.percentile(xyz[:, 2], percentil_piso))
    baixos = xyz[xyz[:, 2] <= corte]
    if len(baixos) < 100:
        raise PlantaError("Pontos insuficientes na faixa do piso para nivelar a nuvem.")

    normal, deslocamento, inliers = ajustar_plano(baixos)
    # Na faixa baixa da nuvem os pes de parede e de movel costumam ser mais
    # numerosos que o proprio piso, entao a fracao esperada e modesta: so abaixo
    # de 10% e sinal de que o plano encontrado nao e o piso.
    fracao = float(inliers.mean())
    if fracao < 0.10:
        avisos.append(
            f"Plano do piso ajustou apenas {fracao:.0%} dos pontos baixos; "
            "a nuvem pode ter mais de um nivel ou muito ruido rasante."
        )
    residuo = float(np.sqrt(np.mean((baixos[inliers] @ normal - deslocamento) ** 2)))
    if residuo > 0.03:
        avisos.append(f"Piso com resíduo RMS de {residuo * 100:.1f} cm; superficie irregular.")

    rotacao = _rotacao_para_vertical(normal)
    desnivel = float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0))))
    if desnivel > 5.0:
        avisos.append(
            f"Nuvem recebida com {desnivel:.1f} graus de desaprumo; "
            "confira a orientacao exportada pelo SLAM GO POST."
        )

    girada = xyz @ rotacao.T
    # Apos girar, o piso fica horizontal: sua altura e a mediana dos inliers.
    z_piso = float(np.median((baixos[inliers] @ rotacao.T)[:, 2]))
    girada[:, 2] -= z_piso
    return girada, rotacao, desnivel, z_piso, avisos


def carregar_nuvem(
    caminho: str | Path,
    voxel: float = 0.02,
    recorte: tuple[float, float, float, float] | None = None,
) -> Nuvem:
    """Le, voxeliza e nivela a nuvem, devolvendo o objeto usado pelas etapas seguintes."""
    arquivo = descobrir_nuvem(caminho)
    centros, pesos, lidos = _ler_blocos(arquivo, voxel, recorte)
    return preparar_nuvem(centros, pesos, voxel, origem=arquivo, pontos_lidos=lidos)


def preparar_nuvem(
    xyz: np.ndarray,
    pesos: np.ndarray | None = None,
    voxel: float = 0.02,
    origem: Path | None = None,
    pontos_lidos: int = 0,
) -> Nuvem:
    """Nivela e recorta um pavimento a partir de pontos ja em memoria.

    Separado de `carregar_nuvem` para que testes e nuvens sinteticas percorram
    exatamente o mesmo caminho de nivelamento e selecao de nivel.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if pesos is None:
        pesos = np.ones(len(xyz), dtype=np.int64)
    pesos = np.asarray(pesos, dtype=np.int64)
    if len(pesos) != len(xyz):
        raise PlantaError("Pesos e pontos com tamanhos diferentes.")

    nivelada, rotacao, desnivel, z_piso, avisos = nivelar(xyz, pesos)
    # Translada para origem local. A nuvem do SLAM GO POST pode vir em
    # coordenadas de projeto (centenas de milhares de metros): manter esses
    # valores adiante gasta digitos significativos em grandezas que a planta
    # mede em centimetros. `origem_mundo` guarda o que foi retirado.
    canto = nivelada[:, :2].min(axis=0)
    origem_mundo = np.array([canto[0], canto[1], z_piso])
    nivelada[:, 0] -= canto[0]
    nivelada[:, 1] -= canto[1]

    z_teto = detectar_teto(nivelada[:, 2], pesos)
    if z_teto < 1.8:
        avisos.append(
            f"Teto estimado em {z_teto:.2f} m, abaixo do esperado para um pavimento; "
            "a nuvem pode nao cobrir a altura toda."
        )

    for nivel in _detectar_niveis(nivelada[:, 2], pesos, z_teto):
        avisos.append(
            f"Massa relevante de pontos em z = {nivel:.2f} m, acima do teto deste "
            "pavimento; recortado (escada ou outro nivel)."
        )

    # Selecao do pavimento: do piso ao teto, com folga para a laje e o rodape.
    dentro = (nivelada[:, 2] >= -0.20) & (nivelada[:, 2] <= z_teto + 0.20)
    if dentro.sum() < 100:
        raise PlantaError("Pavimento selecionado ficou sem pontos; confira o nivelamento.")

    return Nuvem(
        xyz=nivelada[dentro],
        pesos=pesos[dentro],
        voxel=voxel,
        z_teto=z_teto,
        rotacao=rotacao,
        desnivel_graus=desnivel,
        origem_mundo=origem_mundo,
        origem=origem,
        pontos_lidos=pontos_lidos or int(pesos.sum()),
        avisos=avisos,
    )
