r"""Gera a planta baixa em DXF a partir da nuvem registrada do SLAM 100.

Uso, a partir desta pasta:

```powershell
python .\planta_baixa.py ..\SN_000055_certo -o planta_secao_cartografica.dxf --diagnosticos diag\
```

A entrada pode ser a pasta do projeto, que e varrida em busca do `.las`/`.laz`
exportado pelo SLAM GO POST, ou o arquivo direto. Os parametros tem valores de
partida pensados para interior de alvenaria; o relatorio em `--diagnosticos`
traz os numeros que orientam o ajuste de cada um.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from plantabaixa.erros import PlantaError
from plantabaixa.pipeline import Parametros, gerar_planta
from plantabaixa.vaos import Limites


def _recorte(texto: str) -> tuple[float, float, float, float]:
    partes = texto.replace(";", ",").split(",")
    if len(partes) != 4:
        raise argparse.ArgumentTypeError("Use --recorte x0,y0,x1,y1 em coordenadas da nuvem.")
    try:
        x0, y0, x1, y1 = (float(p) for p in partes)
    except ValueError as erro:
        raise argparse.ArgumentTypeError(f"Recorte com valor nao numerico: {texto}") from erro
    if x1 <= x0 or y1 <= y0:
        raise argparse.ArgumentTypeError("No recorte, x1 e y1 devem ser maiores que x0 e y0.")
    return x0, y0, x1, y1


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extrai planta baixa (paredes, portas e janelas) de nuvem de pontos registrada.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("entrada", type=Path, help="Pasta do projeto ou arquivo .las/.laz.")
    parser.add_argument("-o", "--saida", type=Path, default=Path("planta.dxf"),
                        help="Arquivo DXF de saida.")
    parser.add_argument("--diagnosticos", type=Path, default=None,
                        help="Pasta para relatorio e figuras de diagnostico.")
    parser.add_argument("--sobrescrever", action="store_true",
                        help="Permite substituir um DXF existente.")

    grupo = parser.add_argument_group("nuvem")
    grupo.add_argument("--voxel", type=float, default=0.02,
                       help="Aresta do voxel de reducao da nuvem, em metros.")
    grupo.add_argument("--recorte", type=_recorte, default=None,
                       help="Recorte x0,y0,x1,y1 nas coordenadas da nuvem de entrada.")

    grupo = parser.add_argument_group("paredes")
    grupo.add_argument("--res", type=float, default=0.02, help="Resolucao da grade 2D, em metros.")
    grupo.add_argument("--persistencia", type=float, default=0.50,
                       help="Fracao minima do vao piso-teto para a celula valer como parede.")
    grupo.add_argument("--min-parede", type=float, default=0.40,
                       help="Comprimento minimo de face de parede, em metros.")
    grupo.add_argument("--lacuna", type=float, default=0.30,
                       help="Vazio minimo ao longo da face que conta como lacuna, em metros.")
    grupo.add_argument("--tol-face", type=float, default=0.04,
                       help="Tolerancia de distancia da celula ao plano da face, em metros.")
    grupo.add_argument("--tol-angulo", type=float, default=12.0,
                       help="Desvio angular tolerado para a celula pertencer a um eixo, em graus.")
    grupo.add_argument("--espessura-max", type=float, default=0.35,
                       help="Maior separacao entre faces aceita como uma mesma parede, em metros.")
    grupo.add_argument("--espessura-padrao", type=float, default=0.15,
                       help="Espessura adotada para parede vista de um lado so, em metros.")
    grupo.add_argument("--fechamento", type=int, default=1,
                       help="Iteracoes de fechamento morfologico na mascara de parede.")
    grupo.add_argument("--sem-manhattan", action="store_true",
                       help="Desliga a regularizacao por eixos; use em predio nao retilineo.")

    grupo = parser.add_argument_group("vaos")
    grupo.add_argument("--base-porta", type=float, default=0.18,
                       help="Altura maxima da base para o vao ser porta, em metros.")
    grupo.add_argument("--altura-porta", type=float, default=1.60,
                       help="Altura minima de porta, em metros.")
    grupo.add_argument("--largura-porta", type=float, nargs=2, default=(0.55, 1.70),
                       metavar=("MIN", "MAX"), help="Faixa de largura de porta, em metros.")
    grupo.add_argument("--peitoril", type=float, nargs=2, default=(0.40, 1.55),
                       metavar=("MIN", "MAX"), help="Faixa de peitoril de janela, em metros.")
    grupo.add_argument("--nucleo-minimo", type=int, default=15,
                       help="Pontos minimos dentro da espessura para o vao nao ser oclusao.")

    grupo = parser.add_argument_group("desenho")
    grupo.add_argument("--altura-corte", type=float, default=1.40,
                       help="Altura do plano de corte da planta, em metros.")
    grupo.add_argument("--orientacao", choices=("predio", "nuvem", "projeto"), default="predio",
                       help="predio deixa as paredes no esquadro; nuvem preserva a orientacao "
                            "original; projeto acrescenta a origem do levantamento.")
    grupo.add_argument("--sem-eixos", action="store_true", help="Nao desenha a camada EIXOS.")
    grupo.add_argument("--sem-texto", action="store_true", help="Nao desenha a camada TEXTO.")
    grupo.add_argument("--sem-avisos", action="store_true", help="Nao desenha a camada AVISOS.")
    return parser


def parametros_de(args: argparse.Namespace) -> Parametros:
    """Traduz os argumentos da linha de comando nos parametros do pipeline."""
    return Parametros(
        voxel=args.voxel,
        res=args.res,
        persistencia=args.persistencia,
        lacuna=args.lacuna,
        min_parede=args.min_parede,
        tol_face=args.tol_face,
        tol_angulo=args.tol_angulo,
        espessura_max=args.espessura_max,
        espessura_padrao=args.espessura_padrao,
        manhattan=not args.sem_manhattan,
        fechamento=args.fechamento,
        recorte=args.recorte,
        altura_corte=args.altura_corte,
        orientacao=args.orientacao,
        desenhar_eixos=not args.sem_eixos,
        desenhar_texto=not args.sem_texto,
        desenhar_avisos=not args.sem_avisos,
        limites_vaos=Limites(
            base_porta=args.base_porta,
            altura_minima_porta=args.altura_porta,
            largura_minima_porta=args.largura_porta[0],
            largura_maxima_porta=args.largura_porta[1],
            largura_minima_passagem=args.largura_porta[1],
            peitoril_minimo=args.peitoril[0],
            peitoril_maximo=args.peitoril[1],
            pontos_nucleo_minimos=args.nucleo_minimo,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    try:
        resultado = gerar_planta(
            args.entrada,
            saida=args.saida,
            diagnosticos=args.diagnosticos,
            parametros=parametros_de(args),
            sobrescrever=args.sobrescrever,
        )
    except (PlantaError, OSError) as erro:
        print(f"ERRO: {erro}", file=sys.stderr)
        return 1

    print(resultado.relatorio.texto())
    print()
    print(f"Resumo: {resultado.resumo()}")
    if resultado.dxf:
        print(f"DXF gravado: {resultado.dxf}")
    if resultado.diagnosticos:
        print(f"Diagnosticos ({len(resultado.diagnosticos)}): {resultado.diagnosticos[0].parent}")
    if resultado.relatorio.avisos:
        print(f"ATENCAO: {len(resultado.relatorio.avisos)} aviso(s); confira o relatorio.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
