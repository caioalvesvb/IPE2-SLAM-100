"""Verificacao da planificacao contra a nuvem sintetica de planta conhecida.

O que estes testes guardam nao e o numero de segmentos - esse varia com o
RANSAC e com onde a grade cai - e sim as duas propriedades que fazem a
planificacao valer alguma coisa: **todo segmento cai sobre uma face de parede
real** e **nenhum cai sobre mobiliario**. Sem a segunda, um armario de 2 m vira
parede na planta, que e exatamente o modo como esta abordagem falha quando se
remove apenas o piso e o teto.
"""

from __future__ import annotations

import numpy as np
import pytest

import planificar
from tests import sintetico

TOLERANCIA_FACE = 0.05


@pytest.fixture(scope="module")
def las(tmp_path_factory):
    caminho = tmp_path_factory.mktemp("planif") / "sintetica.las"
    sintetico.salvar_las(str(caminho))
    return str(caminho)


def _segmentos(caminho_las, **ajustes):
    """Roda o fluxo completo e devolve (segmentos, xy, diagnostico)."""
    parametros = {
        "voxel": 0.02, "margem_piso": 0.30, "margem_teto": 0.20,
        "persistencia": 0.50, "res": 0.03, "filtrar_moveis": True,
    }
    parametros.update(ajustes)
    xy, _, diagnostico = planificar.planificar(caminho_las, **parametros)

    mascara, origem = planificar.rasterizar(xy, parametros["res"], 5)
    grupos = planificar.aglomerados(mascara, origem, parametros["res"], 15)
    rng = np.random.default_rng(20260827)
    segmentos = []
    for grupo in grupos:
        segmentos += planificar.ajustar_retas(grupo, 0.04, 15, 0.40, 0.30, 300, rng)
    return planificar.fundir_colineares(segmentos, 0.04, 0.40), xy, diagnostico


def _deslocamento(segmentos):
    """Translacao entre o referencial local e o do gabarito.

    Vem da face externa do perimetro, meia espessura fora do eixo da parede.
    """
    xs = np.concatenate([[s.p0[0], s.p1[0]] for s in segmentos])
    ys = np.concatenate([[s.p0[1], s.p1[1]] for s in segmentos])
    return np.array([xs.min(), ys.min()]) + 0.075


def _faces_do_gabarito():
    for parede in sintetico.PAREDES:
        for sinal in (-1, 1):
            yield (parede.eixo, parede.offset + sinal * parede.espessura / 2,
                   parede.inicio, parede.fim, parede.nome)


def _desvio_ate_face(segmento, deslocamento) -> float:
    """Menor distancia do centro do segmento a uma face compativel do gabarito."""
    meio = (segmento.p0 + segmento.p1) / 2 - deslocamento
    direcao = segmento.p1 - segmento.p0
    horizontal = abs(direcao[0]) > abs(direcao[1])
    melhor = float("inf")
    for eixo, offset, inicio, fim, _ in _faces_do_gabarito():
        if (eixo == "x") != horizontal:
            continue
        perpendicular = meio[1] if eixo == "x" else meio[0]
        ao_longo = meio[0] if eixo == "x" else meio[1]
        if not (inicio - 0.3 <= ao_longo <= fim + 0.3):
            continue
        melhor = min(melhor, abs(perpendicular - offset))
    return melhor


def test_nivelamento_e_teto(las):
    _, _, diagnostico = _segmentos(las)
    esperado = np.degrees(np.arccos(np.cos(np.deg2rad(1.5)) * np.cos(np.deg2rad(0.8))))
    assert diagnostico["desnivel_graus"] == pytest.approx(esperado, abs=0.15)
    assert diagnostico["z_teto"] == pytest.approx(sintetico.ALTURA, abs=0.05)


def test_todo_segmento_cai_sobre_parede_real(las):
    segmentos, _, _ = _segmentos(las)
    assert segmentos, "nenhum segmento extraido"
    fora = [s for s in segmentos if _desvio_ate_face(s, _deslocamento(segmentos)) > TOLERANCIA_FACE]
    assert not fora, (
        f"{len(fora)} de {len(segmentos)} segmentos longe de qualquer face: "
        + ", ".join(f"{np.round((s.p0 + s.p1) / 2, 2).tolist()}" for s in fora[:5])
    )


def test_perimetro_tem_as_dimensoes_certas(las):
    segmentos, _, _ = _segmentos(las)
    xs = np.concatenate([[s.p0[0], s.p1[0]] for s in segmentos])
    ys = np.concatenate([[s.p0[1], s.p1[1]] for s in segmentos])
    # Face externa a face externa: eixo a eixo mais uma espessura de parede.
    assert xs.max() - xs.min() == pytest.approx(16.0 + 0.15, abs=0.04)
    assert ys.max() - ys.min() == pytest.approx(10.0 + 0.15, abs=0.04)


def test_filtro_tira_o_mobiliario_alto(las):
    """O armario de 2 m e o caso que derruba a planificacao sem filtro.

    Removendo so piso e teto, ele sobrevive e, projetado em planta, e um
    retangulo denso indistinguivel de uma parede. O teste cobra os dois lados:
    com o filtro, nenhum segmento cai sobre movel; sem ele, cai.
    """
    com_filtro, _, _ = _segmentos(las)
    sem_filtro, _, _ = _segmentos(las, filtrar_moveis=False)

    def sobre_moveis(segmentos):
        deslocamento = _deslocamento(segmentos)
        encontrados = []
        for x0, y0, x1, y1, altura in sintetico.MOVEIS:
            if altura < 1.5:
                continue  # movel baixo ja cai fora pela faixa de altura
            for segmento in segmentos:
                meio = (segmento.p0 + segmento.p1) / 2 - deslocamento
                if x0 - 0.1 <= meio[0] <= x1 + 0.1 and y0 - 0.1 <= meio[1] <= y1 + 0.1:
                    encontrados.append((altura, np.round(meio, 2).tolist()))
        return encontrados

    assert not sobre_moveis(com_filtro), "o filtro deveria ter tirado o mobiliario alto"
    assert sobre_moveis(sem_filtro), (
        "sem filtro o armario precisa aparecer como parede; se nao aparece, "
        "o teste perdeu o caso que ele existe para vigiar"
    )


def test_fusao_nao_junta_as_duas_faces_de_uma_parede():
    """Fundir pedacos colineares nao pode colapsar a espessura da parede."""
    direcao = np.array([1.0, 0.0])
    faces = [
        planificar.Segmento(np.array([0.0, 0.0]), np.array([5.0, 0.0]), 100),
        planificar.Segmento(np.array([0.0, 0.12]), np.array([5.0, 0.12]), 100),
        # Pedaco colinear da primeira face, que *deve* ser fundido.
        planificar.Segmento(np.array([5.2, 0.0]), np.array([8.0, 0.0]), 60),
    ]
    fundidos = planificar.fundir_colineares(faces, tol_lateral=0.04, lacuna=0.40)
    assert len(fundidos) == 2, "as duas faces da parede tem de continuar separadas"
    comprimentos = sorted(s.comprimento for s in fundidos)
    assert comprimentos[0] == pytest.approx(5.0, abs=0.01)
    assert comprimentos[1] == pytest.approx(8.0, abs=0.01), "o pedaco colinear nao foi fundido"


def test_fusao_respeita_vao_de_porta():
    """Dois trechos da mesma reta separados por um vao continuam separados."""
    trechos = [
        planificar.Segmento(np.array([0.0, 0.0]), np.array([3.0, 0.0]), 80),
        planificar.Segmento(np.array([3.9, 0.0]), np.array([7.0, 0.0]), 80),
    ]
    fundidos = planificar.fundir_colineares(trechos, tol_lateral=0.04, lacuna=0.40)
    assert len(fundidos) == 2, "vao de 0,9 m nao pode ser costurado"


def test_cli_roda_e_grava_as_saidas(las, tmp_path):
    dxf = tmp_path / "planta.dxf"
    png = tmp_path / "planta.png"
    plano = tmp_path / "plano.las"
    codigo = planificar.main(
        [las, "--dxf", str(dxf), "--png", str(png), "--las-plano", str(plano)]
    )
    assert codigo == 0
    assert dxf.exists() and png.exists() and plano.exists()

    import laspy

    gravada = laspy.read(str(plano))
    assert len(gravada.points) > 1000
    assert np.abs(np.asarray(gravada.z)).max() < 1e-3, "a nuvem planificada tem de ter z = 0"


def test_margens_incoerentes_viram_erro_de_uso(las):
    from plantabaixa.erros import PlantaError

    with pytest.raises(PlantaError, match="Faixa vazia"):
        planificar.planificar(las, 0.02, 2.0, 2.0, 0.50, 0.03, True)
