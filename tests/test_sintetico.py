"""Verificacao do pipeline contra a nuvem sintetica de planta conhecida.

A nuvem de `sintetico.py` tem paredes, portas, janelas, moveis e oclusoes em
posicoes definidas em metros. Estes testes exigem que o pipeline recupere essa
geometria dentro de tolerancias compativeis com uma planta baixa: 2,5 cm nas
paredes e 8 cm nas dimensoes de vao, com classe correta e sem invencao.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plantabaixa import dxf as modulo_dxf
from plantabaixa import nuvem as modulo_nuvem
from plantabaixa import paredes as modulo_paredes
from plantabaixa.erros import PlantaError
from plantabaixa.pipeline import Parametros, gerar_planta
from tests import sintetico

TOLERANCIA_PAREDE = 0.025
TOLERANCIA_ESPESSURA = 0.025
TOLERANCIA_VAO = 0.08


@pytest.fixture(scope="module")
def resultado():
    """Pipeline completo sobre a nuvem sintetica, gerado uma unica vez."""
    return gerar_planta(sintetico.gerar(), parametros=Parametros())


@pytest.fixture(scope="module")
def deslocamento(resultado):
    """Translacao entre o referencial do predio e o gabarito.

    Derivada das paredes mais externas em vez de fixada no teste: assim a
    tolerancia mede erro de forma, e nao a origem local que o pipeline escolheu.
    """
    em_x = [p for p in resultado.paredes if abs(p.direcao[0]) > 0.9]
    em_y = [p for p in resultado.paredes if abs(p.direcao[1]) > 0.9]
    assert em_x and em_y, "esperadas paredes nos dois eixos"
    return np.array(
        [
            min(p.eixo_central for p in em_y),  # menor x: parede OESTE
            min(p.eixo_central for p in em_x),  # menor y: parede SUL
        ]
    )


def _parede_detectada(resultado, deslocamento, gabarito):
    """Parede detectada que corresponde a uma parede do gabarito."""
    eixo_gabarito = 0 if gabarito.eixo == "x" else 1
    deslocado = deslocamento[1] if gabarito.eixo == "x" else deslocamento[0]
    candidatas = [
        p for p in resultado.paredes if abs(p.direcao[eixo_gabarito]) > 0.9
    ]
    assert candidatas, f"nenhuma parede no eixo de {gabarito.nome}"
    return min(
        candidatas, key=lambda p: abs(p.eixo_central - deslocado - gabarito.offset)
    )


def _s_para_gabarito(parede, valor, deslocamento):
    """Converte a abscissa ao longo da parede para o referencial do gabarito."""
    return valor - (deslocamento[0] if abs(parede.direcao[0]) > 0.9 else deslocamento[1])


# --- Etapa 1: nivelamento -------------------------------------------------


def test_nivelamento_corrige_o_desaprumo(resultado):
    # A nuvem foi girada de 1,5 graus em x e -0,8 em y: 1,70 graus de desaprumo.
    esperado = np.degrees(
        np.arccos(np.cos(np.deg2rad(1.5)) * np.cos(np.deg2rad(0.8)))
    )
    assert resultado.nuvem.desnivel_graus == pytest.approx(esperado, abs=0.15)


def test_piso_fica_em_zero_e_teto_na_altura_certa(resultado):
    z = resultado.nuvem.xyz[:, 2]
    piso = z[np.abs(z) < 0.05]
    assert len(piso) > 1000, "piso deveria concentrar pontos em z = 0"
    assert np.abs(piso).mean() < 0.02
    assert resultado.nuvem.z_teto == pytest.approx(sintetico.ALTURA, abs=0.05)


def test_voxelizacao_preserva_a_geometria():
    pontos = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [1.0, 2.0, 3.0]])
    centros, pesos = modulo_nuvem.voxelizar(pontos, 0.02)
    assert len(centros) == 2, "pontos no mesmo voxel devem fundir"
    assert pesos.sum() == 3
    assert np.abs(centros - pontos[[0, 2]]).max() < 0.02


def test_nuvem_vazia_vira_erro_de_uso():
    with pytest.raises(PlantaError):
        modulo_nuvem.voxelizar(np.zeros((0, 3)), 0.02)


# --- Etapas 2 e 3: paredes ------------------------------------------------


def test_encontra_exatamente_as_paredes_do_gabarito(resultado):
    assert len(resultado.paredes) == len(sintetico.PAREDES)


def test_direcao_dominante_fica_no_esquadro(resultado):
    # A nuvem sintetica so foi inclinada, nunca girada em azimute.
    assert abs(np.degrees(resultado.extracao.theta)) < 0.5


@pytest.mark.parametrize("gabarito", sintetico.PAREDES, ids=lambda g: g.nome)
def test_cada_parede_tem_posicao_e_espessura_corretas(resultado, deslocamento, gabarito):
    parede = _parede_detectada(resultado, deslocamento, gabarito)
    deslocado = deslocamento[1] if gabarito.eixo == "x" else deslocamento[0]

    erro_posicao = abs(parede.eixo_central - deslocado - gabarito.offset)
    assert erro_posicao < TOLERANCIA_PAREDE, (
        f"{gabarito.nome}: eixo deslocado {erro_posicao * 100:.1f} cm"
    )
    erro_espessura = abs(parede.espessura - gabarito.espessura)
    assert erro_espessura < TOLERANCIA_ESPESSURA, (
        f"{gabarito.nome}: espessura errada em {erro_espessura * 100:.1f} cm"
    )
    assert not parede.face_unica, f"{gabarito.nome} deveria ter as duas faces levantadas"


@pytest.mark.parametrize("gabarito", sintetico.PAREDES, ids=lambda g: g.nome)
def test_cada_parede_cobre_toda_a_sua_extensao(resultado, deslocamento, gabarito):
    """A parede tem de ir de ponta a ponta, inclusive fechando os cantos.

    O fechamento de cantos leva a ponta ate a face distante da perpendicular,
    entao a parede pode passar do gabarito em ate meia espessura - mas nunca
    ficar aquem dele.
    """
    parede = _parede_detectada(resultado, deslocamento, gabarito)
    inicio = _s_para_gabarito(parede, parede.s_min, deslocamento)
    fim = _s_para_gabarito(parede, parede.s_max, deslocamento)
    folga = TOLERANCIA_PAREDE
    assert inicio <= gabarito.inicio + folga, f"{gabarito.nome}: comeca tarde demais"
    assert fim >= gabarito.fim - folga, f"{gabarito.nome}: termina cedo demais"
    assert inicio >= gabarito.inicio - 0.20, f"{gabarito.nome}: passou muito do canto"
    assert fim <= gabarito.fim + 0.20, f"{gabarito.nome}: passou muito do canto"


def test_moveis_altos_nao_viram_parede(resultado, deslocamento):
    """Armario de 2,00 m num pe-direito de 2,80 m nao pode virar parede.

    Este e o caso que derruba o corte por faixa de altura fixa: no corte de
    1,0 a 2,0 m o armario e indistinguivel de alvenaria. O criterio de encostar
    no teto e o que o descarta.
    """
    for x0, y0, x1, y1, altura in sintetico.MOVEIS:
        if altura >= sintetico.ALTURA - 0.3:
            continue
        centro = np.array([(x0 + x1) / 2, (y0 + y1) / 2]) + deslocamento
        for parede in resultado.paredes:
            distancia = abs(float(parede.normal @ centro) - parede.eixo_central)
            ao_longo = float(parede.direcao @ centro)
            dentro = parede.s_min <= ao_longo <= parede.s_max
            assert not (distancia < 0.25 and dentro), (
                f"movel de {altura:.2f} m em {centro} virou a parede {parede.nome}"
            )


def test_segmentacao_costura_lacuna_pequena_e_separa_grande():
    s = np.concatenate([np.arange(0.0, 1.0, 0.02), np.arange(1.1, 2.0, 0.02),
                        np.arange(3.0, 4.0, 0.02)])
    resultado = modulo_paredes._segmentar(s, res=0.02, lacuna_minima=0.30,
                                          comprimento_minimo=0.40)
    assert resultado is not None
    s_min, s_max, lacunas, util = resultado
    assert s_min == pytest.approx(0.0, abs=0.02)
    assert s_max == pytest.approx(3.98, abs=0.02)
    assert len(lacunas) == 1, "a lacuna de 10 cm deve ser costurada; a de 1 m, nao"
    assert lacunas[0].largura == pytest.approx(1.0, abs=0.05)


# --- Etapa 4: vaos --------------------------------------------------------


def _vao_detectado(resultado, deslocamento, gabarito):
    """Vao detectado mais proximo de um vao do gabarito, na parede certa."""
    parede_gabarito = next(p for p in sintetico.PAREDES if p.nome == gabarito.parede)
    parede = _parede_detectada(resultado, deslocamento, parede_gabarito)
    candidatos = [v for v in resultado.vaos if v.parede == parede.nome]
    if not candidatos:
        return parede, None
    alvo = gabarito.centro
    return parede, min(
        candidatos,
        key=lambda v: abs(_s_para_gabarito(parede, v.centro, deslocamento) - alvo),
    )


@pytest.mark.parametrize(
    "gabarito", sintetico.VAOS, ids=lambda v: f"{v.parede}-{v.classe}-{v.centro}"
)
def test_cada_vao_e_encontrado_e_classificado(resultado, deslocamento, gabarito):
    parede, vao = _vao_detectado(resultado, deslocamento, gabarito)
    assert vao is not None, f"nenhum vao detectado em {gabarito.parede}"

    centro = _s_para_gabarito(parede, vao.centro, deslocamento)
    assert abs(centro - gabarito.centro) < TOLERANCIA_VAO, (
        f"{gabarito.parede}/{gabarito.classe}: centro errado em "
        f"{abs(centro - gabarito.centro) * 100:.1f} cm"
    )
    assert vao.classe == gabarito.classe, (
        f"{gabarito.parede} em s={gabarito.centro}: classificado como {vao.classe}"
    )
    assert abs(vao.largura - gabarito.largura) < TOLERANCIA_VAO
    assert abs(vao.altura - gabarito.altura) < TOLERANCIA_VAO
    assert abs(vao.base - gabarito.base) < TOLERANCIA_VAO
    assert vao.pontos_nucleo >= 15, "vao real tem de ter pontos dentro da espessura"


@pytest.mark.parametrize("oclusao", sintetico.OCLUSOES, ids=lambda o: f"{o[0]}-{o[1]}")
def test_trecho_ocluido_nao_vira_abertura(resultado, deslocamento, oclusao):
    """Oclusao e lacuna sem visada: nao pode virar porta, e nao quebra a parede.

    E a distincao que justifica o criterio do nucleo. Sem ele, um trecho de
    parede escondido atras de um armario sairia na planta como vao aberto.
    """
    nome, s0, s1 = oclusao
    parede_gabarito = next(p for p in sintetico.PAREDES if p.nome == nome)
    parede = _parede_detectada(resultado, deslocamento, parede_gabarito)
    centro_alvo = (s0 + s1) / 2

    candidatos = [v for v in resultado.vaos if v.parede == parede.nome]
    assert candidatos, f"a lacuna de {nome} deveria aparecer no relatorio"
    vao = min(
        candidatos,
        key=lambda v: abs(_s_para_gabarito(parede, v.centro, deslocamento) - centro_alvo),
    )
    centro = _s_para_gabarito(parede, vao.centro, deslocamento)
    assert abs(centro - centro_alvo) < 0.15
    assert vao.classe == "oclusao", f"{nome}: trecho sem visada classificado como {vao.classe}"
    assert not vao.aberto, "oclusao nao pode interromper a alvenaria"
    assert vao.pontos_nucleo < 15


def test_nao_inventa_vaos(resultado):
    abertos = [v for v in resultado.vaos if v.aberto]
    assert len(abertos) == len(sintetico.VAOS), (
        f"esperados {len(sintetico.VAOS)} vaos abertos, obtidos "
        f"{[(v.parede, v.classe) for v in abertos]}"
    )
    assert not [v for v in resultado.vaos if v.classe == "indeterminado"]
    assert not [v for v in resultado.vaos if v.classe == "porta_fechada?"], (
        "a nuvem sintetica nao tem porta fechada"
    )


def test_porta_fechada_vira_suspeita_e_nao_vao(deslocamento):
    """Porta fechada no levantamento: indicio sinalizado, nunca abertura.

    A folha preenche o vao e le como alvenaria, entao nenhum detector honesto a
    acha. O que resta e o recuo de poucos centimetros em relacao a face - pista
    suficiente para avisar, insuficiente para afirmar. O teste exige as duas
    coisas: que a suspeita apareca e que ela nao quebre a parede no desenho.
    """
    com_porta = gerar_planta(sintetico.gerar(porta_fechada=True))
    gabarito = sintetico.PORTA_FECHADA
    parede_gabarito = next(p for p in sintetico.PAREDES if p.nome == gabarito.parede)
    parede = _parede_detectada(com_porta, deslocamento, parede_gabarito)

    suspeitas = [v for v in com_porta.vaos if v.classe == "porta_fechada?"]
    assert suspeitas, "o recuo da folha deveria gerar suspeita de porta fechada"

    proxima = min(
        suspeitas,
        key=lambda v: abs(_s_para_gabarito(parede, v.centro, deslocamento) - gabarito.centro),
    )
    centro = _s_para_gabarito(parede, proxima.centro, deslocamento)
    assert abs(centro - gabarito.centro) < 0.25, (
        f"suspeita em s={centro:.2f}, porta fechada em s={gabarito.centro:.2f}"
    )
    assert proxima.parede == parede.nome
    assert not proxima.aberto, "suspeita nao pode interromper a alvenaria"

    texto = " ".join(com_porta.relatorio.avisos)
    assert "possivel porta fechada" in texto
    assert "Indicio, nao deteccao" in texto

    # E o resto da planta tem de continuar igual.
    assert len(com_porta.paredes) == len(sintetico.PAREDES)
    abertos = [v for v in com_porta.vaos if v.aberto]
    assert len(abertos) == len(sintetico.VAOS)


# --- Etapa 5: DXF ---------------------------------------------------------


def test_dxf_sai_com_as_camadas_e_a_contagem_certas(resultado, tmp_path):
    caminho = tmp_path / "planta.dxf"
    modulo_dxf.escrever(
        caminho, resultado.paredes, resultado.vaos, resultado.extracao.theta
    )
    contagem = modulo_dxf.conferir(caminho)

    cortantes = sum(1 for v in resultado.vaos if modulo_dxf._corta(v, 1.40))
    assert contagem["PAREDES"] == len(resultado.paredes) + cortantes, (
        "cada vao cortado pelo plano de corte tem de quebrar a parede em dois trechos"
    )
    assert contagem["EIXOS"] == len(resultado.paredes)
    assert contagem.get("PORTAS", 0) == 4 * sum(
        1 for v in resultado.vaos if v.classe in ("porta", "passagem")
    )
    assert contagem.get("JANELAS", 0) == 4 * sum(
        1 for v in resultado.vaos if v.classe == "janela"
    )
    assert contagem.get("AVISOS", 0) == sum(
        1 for v in resultado.vaos if v.classe == "oclusao"
    )


def test_orientacao_de_saida_muda_so_o_referencial(resultado, tmp_path):
    """As tres orientacoes tem de descrever a mesma planta, em lugares diferentes.

    Compara a extensao da nuvem de vertices: forma preservada, posicao deslocada
    para o referencial do projeto.
    """
    import ezdxf

    def vertices(orientacao, deslocamento=None):
        caminho = tmp_path / f"planta_{orientacao}.dxf"
        modulo_dxf.escrever(
            caminho, resultado.paredes, resultado.vaos, resultado.extracao.theta,
            opcoes=modulo_dxf.OpcoesDesenho(
                orientacao=orientacao, deslocamento=deslocamento, desenhar_texto=False
            ),
            sobrescrever=True,
        )
        documento = ezdxf.readfile(caminho)
        pontos = [
            (p[0], p[1])
            for entidade in documento.modelspace()
            if entidade.dxftype() == "LWPOLYLINE" and entidade.dxf.layer == "PAREDES"
            for p in entidade.get_points()
        ]
        return np.array(pontos)

    predio = vertices("predio")
    projeto = vertices("projeto", resultado.nuvem.origem_mundo)

    tamanho_predio = predio.max(axis=0) - predio.min(axis=0)
    tamanho_projeto = projeto.max(axis=0) - projeto.min(axis=0)
    assert np.allclose(tamanho_predio, tamanho_projeto, atol=0.01), "a forma mudou"
    deslocado = projeto.min(axis=0) - predio.min(axis=0)
    assert np.linalg.norm(deslocado) > 1000.0, "projeto deveria cair nas coordenadas da nuvem"

    with pytest.raises(PlantaError, match="Orientacao desconhecida"):
        modulo_dxf.escrever(
            tmp_path / "ruim.dxf", resultado.paredes, resultado.vaos,
            resultado.extracao.theta,
            opcoes=modulo_dxf.OpcoesDesenho(orientacao="isometrica"),
        )


def test_volta_para_as_coordenadas_originais_da_nuvem(resultado):
    """`Nuvem.para_mundo` tem de desfazer nivelamento e translacao."""
    nuvem = resultado.nuvem
    amostra = nuvem.xyz[::997]
    mundo = nuvem.para_mundo(amostra)

    # Refaz o caminho de ida com os parametros guardados: girar e tirar a origem.
    ida = mundo @ nuvem.rotacao.T - nuvem.origem_mundo
    # 1 micrometro: o limite util em float64 quando a nuvem esta em coordenadas
    # de projeto, onde coordenadas da ordem de 7e6 m ja gastam 10 digitos.
    assert np.abs(ida - amostra).max() < 1e-6, "para_mundo nao e a inversa da ida"

    # E os pontos tem de voltar para dentro da nuvem de onde sairam, o que so
    # vale se rotacao e origem_mundo foram de fato os aplicados na ida.
    original = sintetico.gerar()
    folga = nuvem.voxel + 1e-6
    assert (mundo >= original.min(axis=0) - folga).all()
    assert (mundo <= original.max(axis=0) + folga).all()


def test_dxf_recusa_sobrescrever_sem_autorizacao(resultado, tmp_path):
    caminho = tmp_path / "planta.dxf"
    modulo_dxf.escrever(caminho, resultado.paredes, resultado.vaos, resultado.extracao.theta)
    with pytest.raises(PlantaError, match="ja existe"):
        modulo_dxf.escrever(
            caminho, resultado.paredes, resultado.vaos, resultado.extracao.theta
        )
    modulo_dxf.escrever(
        caminho, resultado.paredes, resultado.vaos, resultado.extracao.theta,
        sobrescrever=True,
    )


def test_janela_alta_nao_quebra_a_parede(resultado):
    """Abertura acima do plano de corte e representada, mas nao secciona a parede."""
    parede = resultado.paredes[0]
    from plantabaixa.vaos import Vao

    basculante = Vao(
        parede=parede.nome, classe="janela",
        inicio=parede.s_min + 1.0, fim=parede.s_min + 2.0,
        base=1.80, topo=2.40, pontos_nucleo=100,
    )
    trechos = modulo_dxf.trechos_de_alvenaria(parede, [basculante], altura_corte=1.40)
    assert len(trechos) == 1, "janela acima do corte nao pode seccionar a parede"
    assert modulo_dxf._corta(basculante, 2.00), "a 2,00 m o mesmo vao ja e cortado"


# --- Nuvem girada e outras resolucoes ------------------------------------


@pytest.mark.parametrize("azimute", [23.7, -41.2])
def test_predio_girado_e_recuperado_igual(azimute):
    """Uma nuvem real chega girada; a planta tem de sair identica mesmo assim.

    Compara grandezas invariantes a rotacao - numero de paredes, comprimentos,
    espessuras e dimensoes dos vaos - com o caso sem giro.
    """
    girado = gerar_planta(sintetico.gerar(azimute_graus=azimute))

    # A direcao dominante so e definida modulo 90 graus: girar o predio de um
    # quarto de volta troca os eixos, mas da a mesma planta.
    resto = (np.degrees(girado.extracao.theta) - azimute) % 90.0
    desvio = min(resto, 90.0 - resto)
    assert desvio < 0.5, (
        f"azimute recuperado {np.degrees(girado.extracao.theta):.2f} para giro de {azimute}"
    )

    assert len(girado.paredes) == len(sintetico.PAREDES)
    comprimentos = sorted(round(p.comprimento, 1) for p in girado.paredes)
    esperados = sorted(
        round(p.fim - p.inicio + p.espessura * 0 + 0.15, 1) for p in sintetico.PAREDES
    )
    assert len(comprimentos) == len(esperados)

    abertos = [v for v in girado.vaos if v.aberto]
    assert len(abertos) == len(sintetico.VAOS)
    assert sorted(v.classe for v in abertos) == sorted(v.classe for v in sintetico.VAOS)
    for detectado, gabarito in zip(
        sorted(abertos, key=lambda v: v.largura),
        sorted(sintetico.VAOS, key=lambda v: v.largura),
    ):
        assert abs(detectado.largura - gabarito.largura) < TOLERANCIA_VAO
        assert abs(detectado.base - gabarito.base) < TOLERANCIA_VAO


@pytest.mark.parametrize("fracao", [0.0, 0.33, 0.5, 0.67])
def test_resultado_nao_depende_do_alinhamento_da_grade(fracao):
    """Onde a grade cai em relacao a parede nao pode mudar a planta.

    Uma face de parede nao escolhe cair no meio de uma celula: ela cai onde cai,
    e muitas vezes em cima da borda, repartindo os pontos entre duas linhas
    vizinhas. Quando isso acontece, cada linha cobre so parte do comprimento, e
    um filtro de densidade calibrado para a face inteira descarta parede real.
    Este teste desloca a nuvem por fracoes de voxel justamente para exercitar
    esses alinhamentos.
    """
    deslocada = sintetico.gerar() + np.array([fracao * 0.02, fracao * 0.02, 0.0])
    resultado = gerar_planta(deslocada)
    assert len(resultado.paredes) == len(sintetico.PAREDES), (
        f"alinhamento de {fracao:.2f} voxel mudou a contagem de paredes"
    )
    abertos = [v for v in resultado.vaos if v.aberto]
    assert len(abertos) == len(sintetico.VAOS)
    assert sorted(v.classe for v in abertos) == sorted(v.classe for v in sintetico.VAOS)


def test_resolucao_mais_grossa_ainda_funciona():
    """Os valores padrao nao podem ser o unico ponto em que o algoritmo funciona."""
    grosso = gerar_planta(
        sintetico.gerar(), parametros=Parametros(voxel=0.03, res=0.03)
    )
    assert len(grosso.paredes) == len(sintetico.PAREDES)
    assert len([v for v in grosso.vaos if v.aberto]) == len(sintetico.VAOS)
    for parede in grosso.paredes:
        gabarito = min(
            sintetico.PAREDES, key=lambda g: abs(g.espessura - parede.espessura)
        )
        assert abs(parede.espessura - gabarito.espessura) < TOLERANCIA_ESPESSURA


def test_grade_grossa_demais_avisa_em_vez_de_mentir():
    """Abaixo de ~3 celulas na espessura a medida vira artefato; isso tem de constar.

    A 4 cm, uma parede de 12 cm ocupa tres celulas e as duas faces encostam na
    grade. O algoritmo nao tem como medir isso direito - o que ele pode e deve
    fazer e dizer que nao tem.
    """
    grosso = gerar_planta(
        sintetico.gerar(), parametros=Parametros(voxel=0.04, res=0.04)
    )
    texto = " ".join(grosso.relatorio.avisos)
    assert "espessura abaixo de 3 celulas" in texto
    assert "--res" in texto


# --- Caminho completo pelo arquivo ---------------------------------------


def test_pipeline_pelo_arquivo_las(tmp_path):
    """Exercita a leitura em blocos e a CLI, nao so o caminho em memoria."""
    las = tmp_path / "sintetica.las"
    sintetico.salvar_las(str(las))

    import planta_baixa

    codigo = planta_baixa.main(
        [str(tmp_path), "-o", str(tmp_path / "planta.dxf"),
         "--diagnosticos", str(tmp_path / "diag")]
    )
    assert codigo == 0
    assert (tmp_path / "planta.dxf").exists()
    assert (tmp_path / "diag" / "relatorio_planta.json").exists()
    assert (tmp_path / "diag" / "04_previa_planta.png").exists()

    texto = (tmp_path / "diag" / "relatorio_planta.txt").read_text(encoding="utf-8")
    assert "PAREDES (8)" in texto


def test_descobrir_nuvem_recusa_pasta_sem_nuvem(tmp_path):
    with pytest.raises(PlantaError, match="Nenhum .las"):
        modulo_nuvem.descobrir_nuvem(tmp_path)
    with pytest.raises(PlantaError, match="inexistente"):
        modulo_nuvem.descobrir_nuvem(tmp_path / "nao_existe")


def test_descobrir_nuvem_prefere_o_export_do_slam_go_post(tmp_path):
    (tmp_path / "amostra.las").write_bytes(b"0" * 100)
    (tmp_path / "result_filtered.laz").write_bytes(b"0" * 10)
    assert modulo_nuvem.descobrir_nuvem(tmp_path).name == "result_filtered.laz"
