"""Planta baixa por planificacao: nivela, tira piso e teto, zera z e ajusta retas.

A ideia e a do nome: projetar tudo o que esta entre o piso e o teto num plano
unico (z = 0) e depois aproximar conjuntos de pontos proximos por retas, que sao
as paredes. O que este arquivo acrescenta a essa ideia sao as tres condicoes sem
as quais ela nao funciona na pratica:

1. **Nivelar antes de medir altura.** Uma nuvem de SLAM chega com um ou dois
   graus de desaprumo, e isso basta para esborrachar os picos de piso e de teto
   no histograma de z: numa sala de 16 m, 1,7 grau espalha o piso por quase meio
   metro. Procurar "o pico da metade de baixo" nessa nuvem erra o piso - medido
   na nuvem de ensaio, errava em 70 cm e deslocava a faixa inteira. Depois de
   nivelar, altura volta a significar altura.

2. **Tirar piso e teto, como ja estava previsto.** Se entrassem, ao zerar z
   cobririam a sala toda e nao sobraria contraste nenhum.

3. **Tirar o mobiliario alto, que e o que faltava.** Piso e teto nao sao os
   unicos horizontais do ambiente: um armario de 2 m, visto de cima depois da
   planificacao, e um retangulo denso identico ao de uma parede - e o RANSAC o
   traca como parede. O criterio que separa os dois nao e a altura em que estao,
   e sim quanto do vao piso-teto cada coluna vertical ocupa e se ela encosta no
   teto. Parede atravessa o vao inteiro; armario para antes.

Fluxo:
  1. Le o .las/.laz em blocos, voxeliza e nivela (piso em z = 0).
  2. Mantem a faixa entre piso e teto, com margem de cada lado.
  3. Descarta colunas que nao atravessam o vao (mobiliario) - desligavel.
  4. Zera z e, opcionalmente, grava a nuvem planificada.
  5. Rasteriza numa grade 2D e separa os aglomerados conexos.
  6. RANSAC sequencial de retas + refino por PCA; quebra nas lacunas (porta,
     janela) e funde os pedacos colineares de uma mesma face.
  7. Exporta DXF e PNG de conferencia.

Uso:
  python planificar.py nuvem.las --dxf planta.dxf --png planta.png --las-plano plano.las
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plantabaixa import nuvem as mod_nuvem
from plantabaixa import paredes as mod_paredes
from plantabaixa.erros import PlantaError


@dataclass(eq=False)  # arrays numpy nao comparam por valor; remove() precisa de identidade
class Segmento:
    """Trecho reto de face de parede, no plano z = 0."""

    p0: np.ndarray
    p1: np.ndarray
    n_pontos: int

    @property
    def comprimento(self) -> float:
        return float(np.linalg.norm(self.p1 - self.p0))

    @property
    def direcao(self) -> np.ndarray:
        d = self.p1 - self.p0
        norma = np.linalg.norm(d)
        return d / norma if norma > 1e-12 else np.array([1.0, 0.0])


# --------------------------------------------------------------- planificacao
def planificar(
    caminho: str,
    voxel: float,
    margem_piso: float,
    margem_teto: float,
    persistencia: float,
    res: float,
    filtrar_moveis: bool,
):
    """Da nuvem bruta ao conjunto de pontos planificados (z = 0).

    Devolve `(xy, nuvem, diagnostico)`. `nuvem` carrega o nivelamento aplicado e
    a origem retirada, para quem precisar voltar as coordenadas do levantamento.
    """
    nuvem = mod_nuvem.carregar_nuvem(caminho, voxel=voxel)
    for aviso in nuvem.avisos:
        print(f"  AVISO: {aviso}")

    z = nuvem.xyz[:, 2]
    z0, z1 = margem_piso, nuvem.z_teto - margem_teto
    if z1 <= z0:
        raise PlantaError(
            f"Faixa vazia: piso+{margem_piso:.2f} = {z0:.2f} m nao e menor que "
            f"teto-{margem_teto:.2f} = {z1:.2f} m. Reduza as margens."
        )
    na_faixa = (z >= z0) & (z <= z1)
    if not na_faixa.any():
        raise PlantaError("Nenhum ponto entre o piso e o teto; confira o nivelamento.")

    diagnostico = {
        "z_teto": nuvem.z_teto,
        "desnivel_graus": nuvem.desnivel_graus,
        "faixa": (z0, z1),
        "na_faixa": int(na_faixa.sum()),
        "descartados_moveis": 0,
    }

    if filtrar_moveis:
        # A grade de persistencia mede, por celula, que fracao do vao piso-teto
        # a coluna ocupa e se ela encosta no teto. Tudo o que para antes do teto
        # - armario, estante, divisoria de baia - cai fora aqui.
        grade = mod_paredes.mapa_ocupacao(nuvem.xyz, nuvem.z_teto, res=res, theta=0.0)
        parede = grade.mascara(persistencia)
        ij = np.floor((nuvem.xyz[:, :2] - grade.origem) / res).astype(np.int64)
        np.clip(ij[:, 0], 0, grade.forma[1] - 1, out=ij[:, 0])
        np.clip(ij[:, 1], 0, grade.forma[0] - 1, out=ij[:, 1])
        selecao = na_faixa & parede[ij[:, 1], ij[:, 0]]
        diagnostico["descartados_moveis"] = int(na_faixa.sum() - selecao.sum())
        if not selecao.any():
            raise PlantaError(
                f"Nenhuma coluna passou a persistencia de {persistencia:.2f}. "
                "Reduza --persistencia ou use --sem-filtro-moveis."
            )
    else:
        selecao = na_faixa

    diagnostico["planificados"] = int(selecao.sum())
    return nuvem.xyz[selecao, :2].copy(), nuvem, diagnostico


def salvar_planificada(xy: np.ndarray, nuvem, saida: str, coordenadas: str) -> None:
    """Grava a nuvem planificada em LAS, com z = 0 em todos os pontos."""
    import laspy

    pontos = np.column_stack([xy, np.zeros(len(xy))])
    if coordenadas == "levantamento":
        pontos = nuvem.para_mundo(pontos)

    cabecalho = laspy.LasHeader(point_format=6, version="1.4")
    cabecalho.scales = np.array([0.001, 0.001, 0.001])
    # A origem acompanha os proprios pontos. Herdar o offset do arquivo de
    # entrada deixaria z = 0 a dezenas de metros do offset de z, gastando
    # alcance do inteiro de 32 bits a toa.
    cabecalho.offsets = pontos.min(axis=0)
    las = laspy.LasData(cabecalho)
    las.x, las.y, las.z = pontos[:, 0], pontos[:, 1], pontos[:, 2]
    las.write(saida)


# ---------------------------------------------------------------------- grade
def rasterizar(xy: np.ndarray, res: float, min_pts: int):
    """Grade de ocupacao 2D dos pontos planificados. Devolve (mascara, origem)."""
    origem = xy.min(axis=0) - res
    ij = np.floor((xy - origem) / res).astype(np.int64)
    forma = tuple(int(v) for v in ij.max(axis=0) + 2)
    contagem = np.bincount(
        ij[:, 0] * forma[1] + ij[:, 1], minlength=forma[0] * forma[1]
    )
    mascara = contagem.reshape(forma) >= min_pts
    # Fecha buracos de uma celula vindos de amostragem irregular; o alcance e
    # curto demais para fechar vao de porta, que e o que se quer preservar.
    return ndimage.binary_closing(mascara, structure=np.ones((3, 3))), origem


def aglomerados(mascara: np.ndarray, origem: np.ndarray, res: float, min_celulas: int):
    """Componentes conexos (8-vizinhanca) -> listas de centros de celula."""
    rotulos, n = ndimage.label(mascara, structure=np.ones((3, 3)))
    grupos = []
    for k in range(1, n + 1):
        ij = np.argwhere(rotulos == k)
        if len(ij) >= min_celulas:
            grupos.append((ij + 0.5) * res + origem)
    return grupos


# ---------------------------------------------------------------------- retas
def ajustar_retas(
    pts: np.ndarray,
    tol: float,
    min_inliers: int,
    lacuna: float,
    comp_min: float,
    iteracoes: int,
    rng: np.random.Generator,
) -> list[Segmento]:
    """RANSAC sequencial: melhor reta, refino por PCA, quebra nas lacunas, repete.

    A reta e infinita, mas a parede nao. So saem do jogo os pontos que entraram
    em algum segmento aceito. Removendo todos os inliers da reta infinita, como
    seria natural escrever, uma reta que passa por uma parede leva junto tudo o
    que estiver casualmente alinhado do outro lado da planta, e esses pontos
    desaparecem antes de serem ajustados pela reta a que de fato pertencem.
    """
    segmentos: list[Segmento] = []
    resto = pts
    while len(resto) >= min_inliers:
        melhor = None
        for _ in range(iteracoes):
            a, b = resto[rng.choice(len(resto), 2, replace=False)]
            d = b - a
            norma = np.linalg.norm(d)
            if norma < 1e-6:
                continue
            normal = np.array([-d[1], d[0]]) / norma
            inl = np.abs((resto - a) @ normal) <= tol
            if melhor is None or inl.sum() > melhor.sum():
                melhor = inl
        if melhor is None or melhor.sum() < min_inliers:
            break

        sel = resto[melhor]
        centro = sel.mean(axis=0)
        _, _, vt = np.linalg.svd(sel - centro, full_matrices=False)
        direcao = vt[0]  # direcao de maior variancia = eixo da parede

        indices = np.flatnonzero(melhor)
        t = (sel - centro) @ direcao
        ordem = np.argsort(t)
        t_ord, indices = t[ordem], indices[ordem]
        cortes = np.flatnonzero(np.diff(t_ord) > lacuna) + 1

        usados: list[int] = []
        for trecho, ids in zip(np.split(t_ord, cortes), np.split(indices, cortes)):
            if len(trecho) < 2 or trecho[-1] - trecho[0] < comp_min:
                continue
            segmentos.append(
                Segmento(
                    centro + trecho[0] * direcao,
                    centro + trecho[-1] * direcao,
                    len(trecho),
                )
            )
            usados.extend(int(i) for i in ids)

        if not usados:
            # Nenhum trecho virou segmento: descarta os inliers desta tentativa,
            # senao a proxima volta encontra a mesma reta e o laco nao termina.
            resto = resto[~melhor]
            continue
        consumidos = np.zeros(len(resto), dtype=bool)
        consumidos[np.asarray(usados)] = True
        resto = resto[~consumidos]
    return segmentos


def fundir_colineares(
    segmentos: list[Segmento],
    tol_lateral: float,
    lacuna: float,
    tol_angulo_graus: float = 5.0,
) -> list[Segmento]:
    """Junta pedacos da mesma face que o RANSAC devolveu separados.

    `tol_lateral` tem de ficar bem abaixo da espessura de parede: as duas faces
    de uma mesma parede sao retas distintas e devem continuar distintas, porque
    e a distancia entre elas que da a espessura.
    """
    restantes = sorted(segmentos, key=lambda s: -s.comprimento)
    saida: list[Segmento] = []
    cos_tol = np.cos(np.deg2rad(tol_angulo_graus))

    while restantes:
        base = restantes.pop(0)
        direcao, origem = base.direcao, base.p0
        normal = np.array([-direcao[1], direcao[0]])
        extremos = [0.0, float((base.p1 - origem) @ direcao)]
        pontos = base.n_pontos
        mudou = True
        while mudou:
            mudou = False
            for outro in list(restantes):
                if abs(float(direcao @ outro.direcao)) < cos_tol:
                    continue
                desvio = max(
                    abs(float((outro.p0 - origem) @ normal)),
                    abs(float((outro.p1 - origem) @ normal)),
                )
                if desvio > tol_lateral:
                    continue
                t0 = float((outro.p0 - origem) @ direcao)
                t1 = float((outro.p1 - origem) @ direcao)
                a, b = min(t0, t1), max(t0, t1)
                if a > max(extremos) + lacuna or b < min(extremos) - lacuna:
                    continue  # mesma reta, mas separados por um vao: nao funde
                extremos += [a, b]
                pontos += outro.n_pontos
                restantes.remove(outro)
                mudou = True
        saida.append(
            Segmento(
                origem + min(extremos) * direcao,
                origem + max(extremos) * direcao,
                pontos,
            )
        )
    return saida


# ---------------------------------------------------------------------- saida
def exportar_dxf(segmentos: list[Segmento], saida: str) -> None:
    import ezdxf

    doc = ezdxf.new("R2010", setup=True)
    doc.header["$INSUNITS"] = 6  # metros
    doc.header["$MEASUREMENT"] = 1
    doc.layers.add("PAREDES", color=7)
    msp = doc.modelspace()
    for s in segmentos:
        msp.add_line(tuple(s.p0), tuple(s.p1), dxfattribs={"layer": "PAREDES"})
    doc.saveas(saida)


def exportar_png(xy: np.ndarray, segmentos: list[Segmento], saida: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 12))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(xy), min(len(xy), 300_000), replace=False)
    ax.scatter(xy[idx, 0], xy[idx, 1], s=0.2, c="0.7", linewidths=0)
    for s in segmentos:
        ax.plot([s.p0[0], s.p1[0]], [s.p0[1], s.p1[1]], "r-", lw=1.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{len(segmentos)} segmentos sobre a nuvem planificada")
    fig.savefig(saida, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------- main
def construir_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Planta baixa por planificacao da nuvem (z = 0) e ajuste de retas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("entrada", help="pasta do projeto ou arquivo .las/.laz")
    ap.add_argument("--voxel", type=float, default=0.02, help="aresta do voxel, m")
    ap.add_argument("--margem-piso", type=float, default=0.30,
                    help="altura acima do piso onde a faixa comeca, m")
    ap.add_argument("--margem-teto", type=float, default=0.20,
                    help="distancia abaixo do teto onde a faixa termina, m")
    ap.add_argument("--persistencia", type=float, default=0.50,
                    help="fracao do vao piso-teto para a coluna valer como parede")
    ap.add_argument("--sem-filtro-moveis", action="store_true",
                    help="nao descarta mobiliario alto (mostra o efeito do filtro)")
    ap.add_argument("--res", type=float, default=0.03, help="celula da grade 2D, m")
    ap.add_argument("--min-pts", type=int, default=5, help="pontos minimos por celula")
    ap.add_argument("--tol", type=float, default=0.04, help="distancia maxima ponto-reta, m")
    ap.add_argument("--lacuna", type=float, default=0.40, help="vazio que quebra um segmento, m")
    ap.add_argument("--comp-min", type=float, default=0.30, help="comprimento minimo de segmento, m")
    ap.add_argument("--min-inliers", type=int, default=15, help="celulas minimas por reta")
    ap.add_argument("--sem-fusao", action="store_true",
                    help="nao funde pedacos colineares da mesma face")
    ap.add_argument("--tol-fusao", type=float, default=0.04,
                    help="desvio lateral maximo para fundir dois pedacos, m")
    ap.add_argument("--dxf", help="saida DXF")
    ap.add_argument("--png", help="figura de conferencia")
    ap.add_argument("--las-plano", help="grava a nuvem planificada (z = 0)")
    ap.add_argument("--coordenadas", choices=("local", "levantamento"), default="local",
                    help="referencial do .las planificado")
    return ap


def _milhar(n: int) -> str:
    """Separador de milhar no padrao brasileiro."""
    return f"{n:,}".replace(",", ".")


def main(argv: list[str] | None = None) -> int:
    a = construir_parser().parse_args(argv)
    try:
        xy, nuvem, diag = planificar(
            a.entrada, a.voxel, a.margem_piso, a.margem_teto,
            a.persistencia, a.res, not a.sem_filtro_moveis,
        )
    except (PlantaError, OSError) as erro:
        print(f"ERRO: {erro}", file=sys.stderr)
        return 1

    z0, z1 = diag["faixa"]
    print(f"desaprumo corrigido: {diag['desnivel_graus']:.2f} graus; "
          f"teto a {diag['z_teto']:.2f} m")
    print(f"faixa mantida......: [{z0:.2f}, {z1:.2f}] m acima do piso "
          f"-> {_milhar(diag['na_faixa'])} pontos")
    if diag["descartados_moveis"]:
        print(f"mobiliario alto....: {_milhar(diag['descartados_moveis'])} pontos descartados")
    print(f"planificados.......: {_milhar(diag['planificados'])} pontos")

    if a.las_plano:
        salvar_planificada(xy, nuvem, a.las_plano, a.coordenadas)
        print(f"nuvem planificada..: {a.las_plano}")

    mascara, origem = rasterizar(xy, a.res, a.min_pts)
    grupos = aglomerados(mascara, origem, a.res, a.min_inliers)
    rng = np.random.default_rng(20260827)
    segmentos: list[Segmento] = []
    for grupo in grupos:
        segmentos += ajustar_retas(
            grupo, a.tol, a.min_inliers, a.lacuna, a.comp_min, 300, rng
        )
    brutos = len(segmentos)
    if not a.sem_fusao:
        segmentos = fundir_colineares(segmentos, a.tol_fusao, a.lacuna)
    segmentos.sort(key=lambda s: -s.comprimento)

    print(f"{len(grupos)} aglomerados -> {brutos} segmentos -> {len(segmentos)} apos fusao")
    print(f"comprimento total..: {sum(s.comprimento for s in segmentos):.1f} m")
    print("10 maiores (m).....: " +
          ", ".join(f"{s.comprimento:.2f}" for s in segmentos[:10]))

    if a.dxf:
        exportar_dxf(segmentos, a.dxf)
        print(f"DXF................: {a.dxf}")
    if a.png:
        exportar_png(xy, segmentos, a.png)
        print(f"PNG................: {a.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
