"""Orquestracao das seis etapas, da nuvem registrada ao DXF.

Este modulo nao tem geometria propria: ele so encadeia as etapas e junta o
resultado. Toda decisao tecnica esta em `nuvem`, `paredes`, `vaos` e `dxf`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from plantabaixa import dxf as modulo_dxf
from plantabaixa import nuvem as modulo_nuvem
from plantabaixa import paredes as modulo_paredes
from plantabaixa import qc as modulo_qc
from plantabaixa import vaos as modulo_vaos
from plantabaixa.erros import PlantaError


@dataclass
class Parametros:
    """Tudo o que a linha de comando expoe, num objeto so."""

    voxel: float = 0.02
    res: float = 0.02
    persistencia: float = 0.50
    lacuna: float = 0.30
    min_parede: float = 0.40
    tol_face: float = 0.04
    tol_angulo: float = 12.0
    espessura_max: float = 0.35
    espessura_padrao: float = 0.15
    manhattan: bool = True
    fechamento: int = 1
    recorte: tuple[float, float, float, float] | None = None
    altura_corte: float = 1.40
    orientacao: str = "predio"
    desenhar_eixos: bool = True
    desenhar_texto: bool = True
    desenhar_avisos: bool = True
    limites_vaos: modulo_vaos.Limites = field(default_factory=modulo_vaos.Limites)


@dataclass
class Resultado:
    """Saida completa: geometria, relatorio e caminhos gravados."""

    nuvem: modulo_nuvem.Nuvem
    extracao: modulo_paredes.Extracao
    vaos: list[modulo_vaos.Vao]
    analises: dict[str, modulo_vaos.Analise]
    relatorio: modulo_qc.Relatorio
    dxf: Path | None = None
    diagnosticos: list[Path] = field(default_factory=list)

    @property
    def paredes(self) -> list[modulo_paredes.Parede]:
        return self.extracao.paredes

    def resumo(self) -> str:
        """Uma linha por grandeza principal, para log e para a CLI."""
        classes = self.relatorio.contagem_por_classe()
        descricao = ", ".join(f"{n} {c}" for c, n in sorted(classes.items())) or "nenhum vao"
        return (
            f"{len(self.paredes)} paredes, {descricao}; "
            f"teto a {self.nuvem.z_teto:.2f} m; "
            f"desaprumo corrigido de {self.nuvem.desnivel_graus:.2f} graus; "
            f"giro de esquadro de {self.relatorio.theta_graus:.2f} graus"
        )


def _notas_do_desenho(resultado_relatorio: modulo_qc.Relatorio) -> list[str]:
    """Notas gravadas na camada TEXTO do DXF, para o desenho se explicar sozinho."""
    return [
        "PLANTA BAIXA GERADA AUTOMATICAMENTE DE NUVEM DE PONTOS",
        f"Nuvem: {Path(resultado_relatorio.arquivo).name if resultado_relatorio.arquivo else '(memoria)'}",
        f"Unidades em metros. Piso em z=0; teto a {resultado_relatorio.z_teto:.2f} m.",
        f"Desaprumo corrigido: {resultado_relatorio.desnivel_graus:.3f} graus. "
        f"Giro para esquadro: {resultado_relatorio.theta_graus:.3f} graus.",
        "Camada AVISOS: trecho sem visada ou parede vista de um lado so.",
        "Lado de abertura das portas e hipotese de desenho, nao medida.",
    ]


def gerar_planta(
    entrada: str | Path | np.ndarray,
    saida: str | Path | None = None,
    diagnosticos: str | Path | None = None,
    parametros: Parametros | None = None,
    sobrescrever: bool = False,
) -> Resultado:
    """Gera a planta baixa de uma nuvem registrada.

    `entrada` aceita a pasta do projeto, o arquivo `.las`/`.laz` ou um array
    (N, 3) ja em memoria - este ultimo e o caminho usado pelos testes, que assim
    exercitam exatamente o mesmo pipeline da linha de comando.
    """
    parametros = parametros or Parametros()

    if isinstance(entrada, np.ndarray):
        centros, pesos = modulo_nuvem.voxelizar(entrada, parametros.voxel)
        nuvem = modulo_nuvem.preparar_nuvem(centros, pesos, voxel=parametros.voxel)
    else:
        nuvem = modulo_nuvem.carregar_nuvem(
            entrada, voxel=parametros.voxel, recorte=parametros.recorte
        )

    extracao = modulo_paredes.extrair_paredes(
        nuvem.xyz,
        nuvem.z_teto,
        res=parametros.res,
        persistencia=parametros.persistencia,
        lacuna=parametros.lacuna,
        min_parede=parametros.min_parede,
        tol_face=parametros.tol_face,
        tol_angulo=parametros.tol_angulo,
        espessura_max=parametros.espessura_max,
        espessura_padrao=parametros.espessura_padrao,
        manhattan=parametros.manhattan,
        fechamento=parametros.fechamento,
    )
    # A analise de vaos volta a tocar nos pontos, entao precisa deles no mesmo
    # referencial em que as paredes foram descritas: o do predio.
    vaos, analises, avisos = modulo_vaos.detectar_vaos(
        extracao.paredes, extracao.xyz_predio, nuvem.z_teto, parametros.limites_vaos
    )
    relatorio = modulo_qc.montar_relatorio(nuvem, extracao, vaos, avisos)

    resultado = Resultado(
        nuvem=nuvem,
        extracao=extracao,
        vaos=vaos,
        analises=analises,
        relatorio=relatorio,
    )

    if saida is not None:
        opcoes = modulo_dxf.OpcoesDesenho(
            altura_corte=parametros.altura_corte,
            desenhar_eixos=parametros.desenhar_eixos,
            desenhar_texto=parametros.desenhar_texto,
            desenhar_avisos=parametros.desenhar_avisos,
            orientacao=parametros.orientacao,
            deslocamento=(
                nuvem.origem_mundo if parametros.orientacao == "projeto" else None
            ),
        )
        resultado.dxf = modulo_dxf.escrever(
            saida,
            extracao.paredes,
            vaos,
            extracao.theta,
            notas=_notas_do_desenho(relatorio),
            opcoes=opcoes,
            sobrescrever=sobrescrever,
        )

    if diagnosticos is not None:
        pasta = Path(diagnosticos)
        caminhos = list(modulo_qc.gravar_relatorio(relatorio, pasta))
        caminhos += modulo_qc.gravar_figuras(nuvem, extracao, vaos, analises, pasta)
        resultado.diagnosticos = caminhos

    return resultado


__all__ = ["Parametros", "Resultado", "gerar_planta", "PlantaError"]
