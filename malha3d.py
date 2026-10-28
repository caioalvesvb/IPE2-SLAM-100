"""Planta 3D como malha triangular, a partir da nuvem do SLAM 100.

Dois modos:

  extrusao   Reaproveita o planificar.py: detecta as paredes como segmentos 2D e
             levanta cada uma do piso ao teto como uma caixa (12 triangulos).
             Resultado limpo, leve, tipo maquete - mas so tem paredes retas.
             Objeto que nao chega perto do teto vira uma caixa baixa, entao o
             mobiliario aparece como volume, nao como parede.

  superficie Reconstroi a superficie direto dos pontos: voxeliza a nuvem numa
             grade 3D de ocupacao, suaviza e extrai a isosuperficie por
             Marching Cubes. Pega tudo (moveis, vigas, parede torta), mas a
             malha e pesada e ruidosa.

Duas coisas que a nuvem real do IME obrigou a mudar em relacao ao esboco:

- **A altura so vale depois de nivelar.** A nuvem chega com desaprumo, e o
  histograma de z fica com os picos de piso e teto esborrachados; procurar "o
  pico da metade de baixo" errava o piso em 70 cm. O nivelamento vem do
  `plantabaixa.nuvem`, que e a mesma rotina testada do resto do projeto.

- **A grade do modo superficie sai do conteudo, nao do cabecalho.** Na nuvem do
  IME, 99,9% dos pontos cabem numa caixa de 33 x 27 x 4,5 m, mas 0,1% de
  outliers esticam o bbox para 83 x 85 x 50 m. Dimensionar a grade pelo bbox
  pedia 11,3 GB a 5 cm; pelo conteudo, a mesma resolucao cabe em 136 MB.

Saida em .ply (binario) ou .obj, em metros. Abre no CloudCompare, MeshLab, Blender.

Uso:
  python malha3d.py nuvem.las --modo extrusao   -o planta3d.ply
  python malha3d.py nuvem.las --modo superficie -o superficie.ply --voxel 0.05
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))

import planificar as pl
from plantabaixa import nuvem as mod_nuvem
from plantabaixa import paredes as mod_paredes
from plantabaixa.erros import PlantaError


# ------------------------------------------------------------------- escrita
def salvar_malha(v: np.ndarray, f: np.ndarray, saida: str) -> None:
    """Grava vertices (N,3) e faces (M,3). PLY binario ou OBJ pela extensao."""
    v = np.asarray(v, dtype=np.float64)
    f = np.asarray(f, dtype=np.int32)
    if saida.lower().endswith(".obj"):
        with open(saida, "w") as arq:
            np.savetxt(arq, v, fmt="v %.4f %.4f %.4f")
            np.savetxt(arq, f + 1, fmt="f %d %d %d")
        return
    cabecalho = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(v)}\nproperty double x\nproperty double y\nproperty double z\n"
        f"element face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n"
    )
    # dtype nao alinhado: 1 + 3*4 = 13 bytes por face, que e o que o PLY espera.
    faces = np.zeros(len(f), dtype=[("n", "u1"), ("i", "<i4", 3)])
    faces["n"], faces["i"] = 3, f
    with open(saida, "wb") as arq:
        arq.write(cabecalho.encode())
        arq.write(v.astype("<f8").tobytes())
        arq.write(faces.tobytes())


# ------------------------------------------------------------------ extrusao
# Caixa unitaria: 8 vertices, 12 triangulos com normais para fora.
_CAIXA_F = np.array([
    [0, 2, 1], [0, 3, 2],          # base
    [4, 5, 6], [4, 6, 7],          # topo
    [0, 1, 5], [0, 5, 4],          # lado 0-1
    [1, 2, 6], [1, 6, 5],          # lado 1-2
    [2, 3, 7], [2, 7, 6],          # lado 2-3
    [3, 0, 4], [3, 4, 7],          # lado 3-0
])


def caixa(p0, p1, espessura, z0, z1):
    """Prisma retangular sobre o segmento p0-p1, centrado nele."""
    d = np.asarray(p1, float) - np.asarray(p0, float)
    d = d / np.linalg.norm(d)
    n = np.array([-d[1], d[0]]) * espessura / 2
    base = np.array([p0 - n, p1 - n, p1 + n, p0 + n])
    v = np.vstack([np.c_[base, np.full(4, z0)], np.c_[base, np.full(4, z1)]])
    return v, _CAIXA_F


def laje(xy_min, xy_max, z, espessura):
    """Placa retangular (piso ou teto) cobrindo o retangulo dado."""
    (x0, y0), (x1, y1) = xy_min, xy_max
    base = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    v = np.vstack([np.c_[base, np.full(4, z - espessura)], np.c_[base, np.full(4, z)]])
    return v, _CAIXA_F


def altura_observada(xyz: np.ndarray, s: pl.Segmento, tol: float) -> float:
    """Percentil 99 de z dos pontos a menos de `tol` do segmento.

    O recorte pela caixa envolvente vem antes da projecao porque esta funcao e
    chamada uma vez por segmento: sem ele, cada uma das centenas de chamadas
    percorre os dez milhoes de voxels da nuvem inteira, e a etapa deixa de
    terminar em tempo util.
    """
    lo = np.minimum(s.p0, s.p1) - tol
    hi = np.maximum(s.p0, s.p1) + tol
    caixa = (
        (xyz[:, 0] >= lo[0]) & (xyz[:, 0] <= hi[0])
        & (xyz[:, 1] >= lo[1]) & (xyz[:, 1] <= hi[1])
    )
    if caixa.sum() <= 10:
        return 0.0
    local = xyz[caixa]

    d = s.p1 - s.p0
    comprimento = np.linalg.norm(d)
    u = d / comprimento
    rel = local[:, :2] - s.p0
    t = rel @ u
    perp = np.abs(rel @ np.array([-u[1], u[0]]))
    perto = (perp <= tol) & (t >= 0) & (t <= comprimento)
    return float(np.percentile(local[perto, 2], 99)) if perto.sum() > 10 else 0.0


def modo_extrusao(a) -> tuple[np.ndarray, np.ndarray]:
    nuvem = mod_nuvem.carregar_nuvem(a.entrada, voxel=a.voxel_entrada)
    for aviso in nuvem.avisos:
        print(f"  AVISO: {aviso}")
    teto = nuvem.z_teto
    print(f"nivelado: piso em 0, teto a {teto:.2f} m "
          f"(desaprumo corrigido de {nuvem.desnivel_graus:.2f} graus)")

    z = nuvem.xyz[:, 2]
    na_faixa = (z >= a.margem_piso) & (z <= teto - a.margem_teto)
    xyz = nuvem.xyz[na_faixa]
    if len(xyz) < 1000:
        raise PlantaError("Faixa entre piso e teto quase vazia; confira as margens.")

    # As retas saem so dos pontos de parede. A ideia de deixar o mobiliario
    # entrar e virar caixa baixa e boa no papel e nao sobrevive a um interior
    # real: sobre a nuvem do IME, o RANSAC sem filtro fragmentou a cena em 5.515
    # segmentos - contra 156 com o filtro. Movel nao e reta, e picotar um
    # armario em dezenas de retas curtas nao produz maquete, produz ruido.
    if a.com_moveis:
        base = xyz
    else:
        grade = mod_paredes.mapa_ocupacao(nuvem.xyz, teto, res=a.res, theta=0.0)
        parede = grade.mascara(a.persistencia)
        ij = np.floor((nuvem.xyz[:, :2] - grade.origem) / a.res).astype(np.int64)
        np.clip(ij[:, 0], 0, grade.forma[1] - 1, out=ij[:, 0])
        np.clip(ij[:, 1], 0, grade.forma[0] - 1, out=ij[:, 1])
        base = nuvem.xyz[na_faixa & parede[ij[:, 1], ij[:, 0]]]
        print(f"filtro de moveis: {len(xyz):,} -> {len(base):,} pontos".replace(",", "."))
        if len(base) < 1000:
            raise PlantaError(
                f"Quase nada passou a persistencia de {a.persistencia:.2f}; "
                "reduza --persistencia ou use --com-moveis."
            )

    mascara, origem = pl.rasterizar(base[:, :2], a.res, a.min_pts)
    rng = np.random.default_rng(20260827)
    segmentos = []
    for grupo in pl.aglomerados(mascara, origem, a.res, 15):
        segmentos += pl.ajustar_retas(grupo, 0.04, 15, a.lacuna, a.comp_min, 300, rng)
    if not segmentos:
        raise PlantaError("Nenhuma parede detectada.")
    if not a.sem_fusao:
        segmentos = pl.fundir_colineares(segmentos, 0.04, a.lacuna)

    # Altura de cada segmento = topo dos pontos que caem sobre ele. Assim um
    # movel baixo vira um bloco baixo, e so o que chega perto do teto sobe ate ele.
    partes, n_parede = [], 0
    for s in segmentos:
        topo = altura_observada(xyz, s, tol=0.05)
        if topo >= teto - a.folga_teto:
            topo, n_parede = teto, n_parede + 1
        if topo <= 0.05:
            continue
        partes.append(caixa(s.p0, s.p1, a.espessura, 0.0, topo))
    print(f"{len(segmentos)} segmentos: {n_parede} sobem ate o teto, "
          f"{len(partes) - n_parede} sao volumes baixos")

    if a.lajes:
        pontos = np.vstack([np.vstack([s.p0, s.p1]) for s in segmentos])
        partes.append(laje(pontos.min(axis=0), pontos.max(axis=0), 0.0, 0.10))
        partes.append(laje(pontos.min(axis=0), pontos.max(axis=0), teto + 0.10, 0.10))

    vs, fs, desloc = [], [], 0
    for v, f in partes:
        vs.append(v)
        fs.append(f + desloc)
        desloc += len(v)
    v = np.vstack(vs)
    if a.coordenadas == "levantamento":
        v = nuvem.para_mundo(v)
    return v, np.vstack(fs)


# ---------------------------------------------------------------- superficie
def _extensao_do_conteudo(caminho: str, voxel: float, corte: float):
    """Caixa que contem o grosso dos pontos, ignorando outliers esparsos.

    Dimensionar a grade pelo cabecalho e o que estoura a memoria: numa nuvem de
    SLAM, um punhado de pontos espurios - ceu visto por uma janela, reflexo,
    um lampejo da escada - estica o bbox em dezenas de metros e a grade cresce
    com o cubo disso. Os percentis descrevem onde o predio realmente esta.
    """
    import laspy

    amostras = []
    with laspy.open(caminho) as fonte:
        for bloco in fonte.chunk_iterator(4_000_000):
            xyz = np.column_stack([bloco.x, bloco.y, bloco.z]).astype(np.float64)
            amostras.append(xyz[::10])
    amostra = np.concatenate(amostras)
    lo = np.percentile(amostra, corte, axis=0)
    hi = np.percentile(amostra, 100.0 - corte, axis=0)
    folga = 10 * voxel
    return lo - folga, hi + folga


def modo_superficie(a) -> tuple[np.ndarray, np.ndarray]:
    import laspy
    from skimage.measure import marching_cubes

    origem, limite = _extensao_do_conteudo(a.entrada, a.voxel, a.corte)
    forma = np.ceil((limite - origem) / a.voxel).astype(int) + 2
    mb = forma.prod() * 4 / 1e6
    print(f"conteudo em {np.round(limite - origem, 1)} m -> grade {tuple(forma)} = {mb:.0f} MB")
    if mb > a.max_mb:
        raise PlantaError(
            f"Grade de {mb:.0f} MB acima do limite de {a.max_mb:.0f} MB. "
            "Aumente --voxel, aperte --corte ou suba --max-mb."
        )

    grade = np.zeros(forma, dtype=np.float32)
    plano = np.zeros(forma.prod(), dtype=np.int64)
    dentro_total = 0
    with laspy.open(a.entrada) as fonte:
        for bloco in fonte.chunk_iterator(4_000_000):
            xyz = np.column_stack([bloco.x, bloco.y, bloco.z]).astype(np.float64)
            ijk = np.floor((xyz - origem) / a.voxel).astype(np.int64)
            dentro = np.all((ijk >= 0) & (ijk < forma), axis=1)
            ijk = ijk[dentro]
            dentro_total += len(ijk)
            # bincount no indice achatado em vez de np.add.at, que percorre os
            # pontos um a um e leva minutos em dezenas de milhoes deles.
            achatado = (ijk[:, 0] * forma[1] + ijk[:, 1]) * forma[2] + ijk[:, 2]
            plano += np.bincount(achatado, minlength=plano.size)
    grade = plano.reshape(forma).astype(np.float32)
    print(f"{dentro_total:,} pontos dentro da grade".replace(",", "."))

    # Ocupacao: voxel com pontos suficientes. Fechamento tapa furos de
    # amostragem; o gaussiano arredonda a escada de voxels antes do Marching Cubes.
    ocupado = grade >= a.min_pts
    print(f"{ocupado.sum():,} voxels ocupados".replace(",", "."))
    ocupado = ndimage.binary_closing(ocupado, iterations=1)
    campo = ndimage.gaussian_filter(ocupado.astype(np.float32), sigma=a.sigma)
    # Nivel 0.3: abaixo de 0.5 para que a parede (2-3 voxels de espessura) nao
    # suma depois do gaussiano.
    v, f, _, _ = marching_cubes(campo, level=a.nivel, spacing=(a.voxel,) * 3)
    v = v + origem
    print(f"{len(v):,} vertices, {len(f):,} triangulos".replace(",", "."))
    return v, f


# ----------------------------------------------------------------------- main
def construir_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Malha 3D da nuvem: extrusao das paredes ou isosuperficie.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("entrada", help=".las ou .laz, ou a pasta do projeto")
    ap.add_argument("--modo", choices=("extrusao", "superficie"), default="extrusao")
    ap.add_argument("-o", "--saida", default="planta3d.ply", help=".ply ou .obj")

    g = ap.add_argument_group("extrusao")
    g.add_argument("--voxel-entrada", type=float, default=0.02, help="voxel de leitura, m")
    g.add_argument("--margem-piso", type=float, default=0.10)
    g.add_argument("--margem-teto", type=float, default=0.10)
    g.add_argument("--res", type=float, default=0.03)
    g.add_argument("--min-pts", type=int, default=5)
    g.add_argument("--persistencia", type=float, default=0.50,
                   help="fracao do vao piso-teto para a coluna valer como parede")
    g.add_argument("--com-moveis", action="store_true",
                   help="nao filtra mobiliario: cada objeto vira caixa baixa. "
                        "Em interior mobiliado rende milhares de fragmentos.")
    g.add_argument("--lacuna", type=float, default=0.40)
    g.add_argument("--comp-min", type=float, default=0.30)
    g.add_argument("--sem-fusao", action="store_true")
    g.add_argument("--espessura", type=float, default=0.12, help="espessura da caixa de parede, m")
    g.add_argument("--lajes", action="store_true", help="inclui piso e teto (retangulo envolvente)")
    g.add_argument("--folga-teto", type=float, default=0.30,
                   help="segmento cujo topo chega a essa distancia do teto vira parede inteira, m")
    g.add_argument("--coordenadas", choices=("local", "levantamento"), default="local")

    g = ap.add_argument_group("superficie")
    g.add_argument("--voxel", type=float, default=0.05)
    g.add_argument("--corte", type=float, default=0.2,
                   help="percentil descartado em cada ponta ao dimensionar a grade")
    g.add_argument("--min-pts-voxel", type=int, default=2, dest="min_pts_voxel",
                   help="pontos minimos para o voxel contar como ocupado")
    g.add_argument("--sigma", type=float, default=1.0, help="suavizacao, em voxels")
    g.add_argument("--nivel", type=float, default=0.3, help="isovalor do Marching Cubes")
    g.add_argument("--max-mb", type=float, default=2000)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = construir_parser().parse_args(argv)
    if a.modo == "superficie":
        a.min_pts = a.min_pts_voxel
    try:
        v, f = modo_extrusao(a) if a.modo == "extrusao" else modo_superficie(a)
    except (PlantaError, OSError) as erro:
        print(f"ERRO: {erro}", file=sys.stderr)
        return 1
    salvar_malha(v, f, a.saida)
    print(f"gravado {a.saida} ({len(v):,} vertices, {len(f):,} triangulos)".replace(",", "."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
