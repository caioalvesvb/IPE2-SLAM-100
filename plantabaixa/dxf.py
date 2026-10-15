"""Etapa 5: escrita do DXF em camadas.

O desenho segue a convencao de planta baixa: o corte horizontal passa a
`altura_corte` (1,40 m por padrao) e so interrompe a alvenaria nas aberturas que
o corte atravessa. Uma janela de peitoril 0,95 m e verga 2,25 m e cortada e
aparece como vao com as linhas de peitoril; um basculante alto, que comeca acima
do corte, nao interrompe a parede e e desenhado tracejado - que e como um
projetista o representa.

As paredes saem como polilinhas fechadas, uma por trecho entre aberturas, para
que a abertura seja lacuna de verdade no CAD e nao um risco por cima da parede.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from plantabaixa.erros import PlantaError
from plantabaixa.paredes import Parede, rotacao_2d
from plantabaixa.vaos import Vao

# Camada: (cor ACI, tipo de linha, descricao)
CAMADAS = {
    "PAREDES": (7, "CONTINUOUS", "Corpo das paredes, polilinha fechada por trecho"),
    "EIXOS": (8, "CENTER", "Eixo de cada parede"),
    "PORTAS": (3, "CONTINUOUS", "Vaos de porta e passagem, com folha e arco"),
    "JANELAS": (5, "CONTINUOUS", "Vaos de janela, com peitoril"),
    "VAOS_ALTOS": (4, "DASHED", "Aberturas acima do plano de corte"),
    "AVISOS": (1, "DASHED", "Trechos ocluidos, parede de face unica, suspeitas"),
    "TEXTO": (2, "CONTINUOUS", "Rotulos de vaos e notas do levantamento"),
}


@dataclass
class OpcoesDesenho:
    """Ajustes de representacao do DXF."""

    altura_corte: float = 1.40
    altura_texto: float = 0.12
    desenhar_eixos: bool = True
    desenhar_texto: bool = True
    desenhar_avisos: bool = True
    orientacao: str = "predio"
    # Origem a recolocar quando `orientacao` e "projeto"; vem de Nuvem.origem_mundo.
    deslocamento: np.ndarray | None = None


ORIENTACOES = ("predio", "nuvem", "projeto")


def _transformacao(theta: float, orientacao: str, deslocamento: np.ndarray | None = None):
    """Funcao que leva coordenadas do referencial do predio ao de saida.

    - `predio`: paredes paralelas aos eixos do papel, que e o que se espera de
      uma planta, com a origem onde o pipeline a colocou;
    - `nuvem`: desfaz o giro de esquadro e devolve o desenho na orientacao
      original da nuvem, util para sobrepor ao levantamento;
    - `projeto`: a orientacao da nuvem mais a origem que o nivelamento retirou,
      para o desenho cair nas coordenadas do projeto.

    Uma ressalva sobre `projeto`: a planta vive no referencial *nivelado*, entao
    recolocar a origem nao desfaz o giro de nivelamento. A diferenca em relacao
    ao XY original da nuvem e da ordem de `sen(desaprumo) x altura` - poucos
    centimetros num desaprumo tipico, mas nao zero.
    """
    if orientacao not in ORIENTACOES:
        raise PlantaError(
            f"Orientacao desconhecida: {orientacao}; use {', '.join(ORIENTACOES)}."
        )
    matriz = np.eye(2) if orientacao == "predio" else rotacao_2d(theta).T
    offset = np.zeros(2) if deslocamento is None else np.asarray(deslocamento, dtype=float)[:2]

    def aplicar(xy) -> np.ndarray:
        return np.atleast_2d(np.asarray(xy, dtype=float)) @ matriz + offset

    return aplicar


def _vaos_da_parede(parede: Parede, vaos: list[Vao]) -> list[Vao]:
    return [v for v in vaos if v.parede == parede.nome]


def _corta(vao: Vao, altura_corte: float) -> bool:
    """Se a abertura e atravessada pelo plano de corte da planta."""
    return vao.aberto and vao.base < altura_corte < vao.topo


def trechos_de_alvenaria(
    parede: Parede, vaos: list[Vao], altura_corte: float
) -> list[tuple[float, float]]:
    """Intervalos de `s` em que a parede tem alvenaria no plano de corte."""
    cortantes = sorted(
        (v for v in _vaos_da_parede(parede, vaos) if _corta(v, altura_corte)),
        key=lambda v: v.inicio,
    )
    trechos: list[tuple[float, float]] = []
    inicio = parede.s_min
    for vao in cortantes:
        if vao.inicio > inicio:
            trechos.append((inicio, min(vao.inicio, parede.s_max)))
        inicio = max(inicio, vao.fim)
    if inicio < parede.s_max:
        trechos.append((inicio, parede.s_max))
    return [(a, b) for a, b in trechos if b - a > 1e-6]


def _retangulo(parede: Parede, s_a: float, s_b: float) -> np.ndarray:
    """Quatro cantos do trecho de parede, no referencial do predio."""
    meia = parede.espessura / 2
    base = parede.normal * parede.eixo_central
    return np.array(
        [
            base + parede.normal * meia + parede.direcao * s_a,
            base + parede.normal * meia + parede.direcao * s_b,
            base - parede.normal * meia + parede.direcao * s_b,
            base - parede.normal * meia + parede.direcao * s_a,
        ]
    )


def _angulo(vetor: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(vetor[1], vetor[0]))) % 360.0


def _desenhar_porta(msp, parede: Parede, vao: Vao, para_saida, camada: str) -> None:
    """Ombreiras, folha e arco de abertura.

    A folha sai da ombreira do lado em que o vao tem mais piso levantado; o arco
    vai da ponta da folha ate a outra ombreira, cobrindo os 90 graus do giro.
    """
    meia = parede.espessura / 2
    eixo = parede.normal * parede.eixo_central
    for s in (vao.inicio, vao.fim):
        extremos = para_saida(
            np.array(
                [
                    eixo + parede.normal * meia + parede.direcao * s,
                    eixo - parede.normal * meia + parede.direcao * s,
                ]
            )
        )
        msp.add_line(extremos[0], extremos[1], dxfattribs={"layer": camada})

    dobradica = eixo + parede.direcao * vao.inicio - parede.normal * vao.lado * meia
    abertura = parede.normal * vao.lado
    ponta = dobradica + abertura * vao.largura
    pontos = para_saida(np.array([dobradica, ponta]))
    msp.add_line(pontos[0], pontos[1], dxfattribs={"layer": camada})

    centro = para_saida(dobradica)[0]
    inicio = _angulo(para_saida(abertura)[0] - para_saida(np.zeros(2))[0])
    fim = _angulo(para_saida(parede.direcao)[0] - para_saida(np.zeros(2))[0])
    # O arco do ezdxf vai do inicio ao fim no sentido anti-horario; a ordem e
    # escolhida para cobrir o quadrante do giro, nao os outros tres.
    if (fim - inicio) % 360.0 > 180.0:
        inicio, fim = fim, inicio
    msp.add_arc(
        center=centro,
        radius=vao.largura,
        start_angle=inicio,
        end_angle=fim,
        dxfattribs={"layer": camada},
    )


def _desenhar_janela(msp, parede: Parede, vao: Vao, para_saida, camada: str) -> None:
    """Peitoril atravessando o vao, nas duas faces, mais as ombreiras."""
    meia = parede.espessura / 2
    eixo = parede.normal * parede.eixo_central
    for sinal in (1.0, -1.0):
        linha = para_saida(
            np.array(
                [
                    eixo + parede.normal * sinal * meia + parede.direcao * vao.inicio,
                    eixo + parede.normal * sinal * meia + parede.direcao * vao.fim,
                ]
            )
        )
        msp.add_line(linha[0], linha[1], dxfattribs={"layer": camada})
    for s in (vao.inicio, vao.fim):
        extremos = para_saida(
            np.array(
                [
                    eixo + parede.normal * meia + parede.direcao * s,
                    eixo - parede.normal * meia + parede.direcao * s,
                ]
            )
        )
        msp.add_line(extremos[0], extremos[1], dxfattribs={"layer": camada})


def _desenhar_marca(msp, parede: Parede, vao: Vao, para_saida, camada: str) -> None:
    """Retangulo tracejado para abertura que o corte nao atravessa ou aviso."""
    cantos = para_saida(_retangulo(parede, vao.inicio, vao.fim))
    msp.add_lwpolyline(cantos, close=True, dxfattribs={"layer": camada})


def _rotulo(vao: Vao) -> str:
    if vao.classe == "porta":
        return f"P {vao.largura:.2f}x{vao.altura:.2f}"
    if vao.classe == "passagem":
        return f"PASS {vao.largura:.2f}x{vao.altura:.2f}"
    if vao.classe == "janela":
        return f"J {vao.largura:.2f}x{vao.altura:.2f} h={vao.base:.2f}"
    if vao.classe == "oclusao":
        return f"OCLUSAO {vao.largura:.2f}"
    if vao.classe == "porta_fechada?":
        return f"PORTA FECHADA? {vao.largura:.2f}"
    return f"{vao.classe.upper()} {vao.largura:.2f}x{vao.altura:.2f}"


def escrever(
    caminho: str | Path,
    paredes: list[Parede],
    vaos: list[Vao],
    theta: float,
    notas: list[str] | None = None,
    opcoes: OpcoesDesenho | None = None,
    sobrescrever: bool = False,
) -> Path:
    """Grava o DXF e devolve o caminho efetivamente escrito."""
    import ezdxf

    opcoes = opcoes or OpcoesDesenho()
    caminho = Path(caminho)
    if caminho.exists() and not sobrescrever:
        raise PlantaError(
            f"{caminho} ja existe. Use --sobrescrever ou escolha outro nome de saida."
        )

    documento = ezdxf.new("R2010", setup=True)
    documento.header["$INSUNITS"] = 6  # metros
    documento.header["$MEASUREMENT"] = 1  # sistema metrico
    for nome, (cor, tipo, descricao) in CAMADAS.items():
        camada = documento.layers.add(nome, color=cor, linetype=tipo)
        camada.description = descricao
    msp = documento.modelspace()

    para_saida = _transformacao(theta, opcoes.orientacao, opcoes.deslocamento)

    for parede in paredes:
        for s_a, s_b in trechos_de_alvenaria(parede, vaos, opcoes.altura_corte):
            cantos = para_saida(_retangulo(parede, s_a, s_b))
            msp.add_lwpolyline(cantos, close=True, dxfattribs={"layer": "PAREDES"})
        if opcoes.desenhar_eixos:
            eixo = para_saida(
                np.array(
                    [
                        parede.normal * parede.eixo_central + parede.direcao * parede.s_min,
                        parede.normal * parede.eixo_central + parede.direcao * parede.s_max,
                    ]
                )
            )
            msp.add_line(eixo[0], eixo[1], dxfattribs={"layer": "EIXOS"})
        if opcoes.desenhar_avisos and parede.face_unica:
            cantos = para_saida(_retangulo(parede, parede.s_min, parede.s_max))
            msp.add_lwpolyline(cantos, close=True, dxfattribs={"layer": "AVISOS"})

    por_nome = {p.nome: p for p in paredes}
    for vao in vaos:
        parede = por_nome.get(vao.parede)
        if parede is None:
            continue
        if vao.classe in ("porta", "passagem"):
            camada = "PORTAS"
            if _corta(vao, opcoes.altura_corte):
                _desenhar_porta(msp, parede, vao, para_saida, camada)
            else:
                camada = "VAOS_ALTOS"
                _desenhar_marca(msp, parede, vao, para_saida, camada)
        elif vao.classe == "janela":
            camada = "JANELAS"
            if _corta(vao, opcoes.altura_corte):
                _desenhar_janela(msp, parede, vao, para_saida, camada)
            else:
                camada = "VAOS_ALTOS"
                _desenhar_marca(msp, parede, vao, para_saida, camada)
        else:
            if not opcoes.desenhar_avisos:
                continue
            camada = "AVISOS"
            _desenhar_marca(msp, parede, vao, para_saida, camada)

        if opcoes.desenhar_texto:
            meio = (
                parede.normal * parede.eixo_central
                + parede.direcao * vao.centro
                + parede.normal * vao.lado * (parede.espessura / 2 + 3 * opcoes.altura_texto)
            )
            posicao = para_saida(meio)[0]
            texto = msp.add_text(
                _rotulo(vao),
                height=opcoes.altura_texto,
                dxfattribs={"layer": "TEXTO"},
            )
            texto.set_placement(tuple(posicao))

    if notas and opcoes.desenhar_texto and paredes:
        todos = np.concatenate(
            [para_saida(_retangulo(p, p.s_min, p.s_max)) for p in paredes]
        )
        canto = todos.min(axis=0)
        for i, nota in enumerate(notas):
            texto = msp.add_text(
                nota, height=opcoes.altura_texto, dxfattribs={"layer": "TEXTO"}
            )
            texto.set_placement(
                (float(canto[0]), float(canto[1]) - (i + 2) * opcoes.altura_texto * 2.0)
            )

    caminho.parent.mkdir(parents=True, exist_ok=True)
    documento.saveas(caminho)
    return caminho


def conferir(caminho: str | Path) -> dict[str, int]:
    """Reabre o DXF e conta entidades por camada, para validar a escrita."""
    import ezdxf

    documento = ezdxf.readfile(caminho)
    contagem: dict[str, int] = {}
    for entidade in documento.modelspace():
        camada = entidade.dxf.layer
        contagem[camada] = contagem.get(camada, 0) + 1
    return contagem
