"""Etapa 6: relatorio e figuras de diagnostico.

O produto de entrega e o DXF; o que esta aqui existe para duas coisas. Primeiro,
auditar o resultado: residuo das faces, espessuras medidas, lista de vaos com
dimensoes e o inventario de avisos - trecho ocluido, parede vista de um lado so,
suspeita de porta fechada. Segundo, permitir ajustar os parametros com evidencia
em vez de tentativa: o histograma de `score` mostra onde cai o limiar de
persistencia, e a mascara com as retas sobrepostas mostra o que a vetorizacao
viu.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from plantabaixa.nuvem import Nuvem
from plantabaixa.paredes import Extracao, Parede
from plantabaixa.vaos import Analise, Vao


@dataclass
class Relatorio:
    """Numeros e avisos do levantamento, serializaveis em JSON."""

    arquivo: str | None
    pontos_lidos: int
    voxels: int
    voxel: float
    desnivel_graus: float
    z_teto: float
    theta_graus: float
    paredes: list[dict] = field(default_factory=list)
    vaos: list[dict] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)
    ajustes_canto: list[str] = field(default_factory=list)
    histograma_score: list[dict] = field(default_factory=list)

    def contagem_por_classe(self) -> dict[str, int]:
        contagem: dict[str, int] = {}
        for vao in self.vaos:
            contagem[vao["classe"]] = contagem.get(vao["classe"], 0) + 1
        return contagem

    def texto(self) -> str:
        """Relatorio legivel, no formato que a CLI imprime e grava em .txt."""
        linhas = [
            "RELATORIO DA PLANTA BAIXA",
            "=" * 60,
            f"Nuvem...............: {self.arquivo or '(em memoria)'}",
            f"Pontos lidos........: {self.pontos_lidos:,}".replace(",", "."),
            f"Voxels de {self.voxel * 100:.0f} cm......: {self.voxels:,}".replace(",", "."),
            f"Desaprumo corrigido.: {self.desnivel_graus:.3f} graus",
            f"Altura piso-teto....: {self.z_teto:.3f} m",
            f"Giro p/ esquadro....: {self.theta_graus:.3f} graus",
            "",
            f"PAREDES ({len(self.paredes)})",
            "-" * 60,
        ]
        for parede in self.paredes:
            marca = " [FACE UNICA]" if parede["face_unica"] else ""
            linhas.append(
                f"  {parede['nome']}  compr {parede['comprimento']:6.2f} m  "
                f"esp {parede['espessura'] * 100:5.1f} cm  "
                f"residuo {parede['residuo_mm']:4.1f} mm  "
                f"{parede['celulas']:6d} celulas{marca}"
            )
        contagem = self.contagem_por_classe()
        linhas += [
            "",
            f"VAOS ({len(self.vaos)}): "
            + ", ".join(f"{n} {c}" for c, n in sorted(contagem.items()))
            + ("" if self.vaos else "nenhum"),
            "-" * 60,
        ]
        for vao in self.vaos:
            linhas.append(
                f"  {vao['parede']}  {vao['classe']:14}  "
                f"{vao['largura']:.2f} x {vao['altura']:.2f} m  "
                f"base {vao['base']:.2f} m  nucleo {vao['pontos_nucleo']:5d}"
                + (f"  ({vao['observacao']})" if vao["observacao"] else "")
            )
        if self.avisos:
            linhas += ["", f"AVISOS ({len(self.avisos)})", "-" * 60]
            linhas += [f"  - {aviso}" for aviso in self.avisos]
        if self.ajustes_canto:
            linhas += ["", f"CANTOS FECHADOS ({len(self.ajustes_canto)})", "-" * 60]
            linhas += [f"  - {ajuste}" for ajuste in self.ajustes_canto]
        return "\n".join(linhas)


def _residuo_parede(parede: Parede) -> float:
    """Residuo RMS das faces da parede, em milimetros."""
    if not parede.faces:
        return float("nan")
    quadrados = [face.residuo**2 * face.celulas for face in parede.faces]
    celulas = sum(face.celulas for face in parede.faces)
    return float(np.sqrt(sum(quadrados) / max(1, celulas)) * 1000.0)


def montar_relatorio(
    nuvem: Nuvem,
    extracao: Extracao,
    vaos: list[Vao],
    avisos_vaos: list[str],
) -> Relatorio:
    """Reune os numeros das quatro primeiras etapas num relatorio unico."""
    score = extracao.grade.score[extracao.grade.score > 0]
    bordas = np.arange(0.0, 1.0001, 0.1)
    contagem, _ = np.histogram(score, bins=bordas)
    histograma = [
        {"de": round(float(bordas[i]), 2), "ate": round(float(bordas[i + 1]), 2),
         "celulas": int(contagem[i])}
        for i in range(len(contagem))
    ]

    paredes = [
        {
            "nome": p.nome,
            "comprimento": round(p.comprimento, 3),
            "espessura": round(p.espessura, 3),
            "face_unica": p.face_unica,
            "faces": len(p.faces),
            "celulas": sum(f.celulas for f in p.faces),
            "residuo_mm": round(_residuo_parede(p), 2),
            "eixo_central": round(p.eixo_central, 3),
            "s_min": round(p.s_min, 3),
            "s_max": round(p.s_max, 3),
            "direcao": [round(float(x), 6) for x in p.direcao],
        }
        for p in extracao.paredes
    ]
    registros = []
    for vao in vaos:
        registro = asdict(vao)
        registro["largura"] = round(vao.largura, 3)
        registro["altura"] = round(vao.altura, 3)
        registro["centro"] = round(vao.centro, 3)
        for chave in ("inicio", "fim", "base", "topo", "preenchimento"):
            registro[chave] = round(float(registro[chave]), 3)
        registros.append(registro)

    avisos = list(nuvem.avisos) + list(extracao.avisos) + list(avisos_vaos)
    unicas = [p for p in extracao.paredes if p.face_unica]
    if unicas:
        avisos.append(
            f"{len(unicas)} parede(s) vista(s) de um lado so "
            f"({', '.join(p.nome for p in unicas)}); espessura adotada por padrao."
        )
    obliquas = [p for p in extracao.paredes if p.faces and p.faces[0].eixo < 0]
    if obliquas:
        avisos.append(
            f"{len(obliquas)} parede(s) fora de esquadro "
            f"({', '.join(p.nome for p in obliquas)}); conferir no DXF."
        )
    ocluidos = [v for v in vaos if v.classe == "oclusao"]
    if ocluidos:
        avisos.append(
            f"{len(ocluidos)} trecho(s) de parede sem visada, desenhados continuos "
            "na camada AVISOS."
        )
    indeterminados = [v for v in vaos if v.classe == "indeterminado"]
    if indeterminados:
        avisos.append(
            f"{len(indeterminados)} vao(s) sem classe definida; conferir dimensoes no relatorio."
        )

    return Relatorio(
        arquivo=str(nuvem.origem) if nuvem.origem else None,
        pontos_lidos=int(nuvem.pontos_lidos),
        voxels=int(len(nuvem.xyz)),
        voxel=nuvem.voxel,
        desnivel_graus=round(nuvem.desnivel_graus, 4),
        z_teto=round(nuvem.z_teto, 4),
        theta_graus=round(float(np.degrees(extracao.theta)), 4),
        paredes=paredes,
        vaos=registros,
        avisos=avisos,
        ajustes_canto=list(extracao.ajustes_canto),
        histograma_score=histograma,
    )


def gravar_relatorio(relatorio: Relatorio, pasta: str | Path) -> tuple[Path, Path]:
    """Grava `relatorio_planta.json` e `.txt` na pasta de diagnosticos."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    caminho_json = pasta / "relatorio_planta.json"
    caminho_txt = pasta / "relatorio_planta.txt"
    caminho_json.write_text(
        json.dumps(asdict(relatorio), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    caminho_txt.write_text(relatorio.texto(), encoding="utf-8")
    return caminho_json, caminho_txt


def gravar_figuras(
    nuvem: Nuvem,
    extracao: Extracao,
    vaos: list[Vao],
    analises: dict[str, Analise],
    pasta: str | Path,
    max_paredes_sh: int = 8,
) -> list[Path]:
    """Gera as figuras de diagnostico e devolve os caminhos gravados."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    gravadas: list[Path] = []

    # 1. Histograma de altura: onde estao piso e teto.
    figura, eixo = plt.subplots(figsize=(7, 4))
    eixo.hist(nuvem.xyz[:, 2], bins=120, weights=nuvem.pesos, color="tab:blue")
    eixo.axvline(0.0, color="tab:green", lw=1.2, label="piso (z = 0)")
    eixo.axvline(nuvem.z_teto, color="tab:red", lw=1.2, label=f"teto ({nuvem.z_teto:.2f} m)")
    eixo.set_xlabel("altura (m)")
    eixo.set_ylabel("pontos")
    eixo.set_title("Distribuicao de altura apos o nivelamento")
    eixo.legend()
    figura.tight_layout()
    caminho = pasta / "01_histograma_altura.png"
    figura.savefig(caminho, dpi=130)
    plt.close(figura)
    gravadas.append(caminho)

    # 2. Mapa de persistencia vertical: a medida que separa parede de movel.
    figura, eixo = plt.subplots(figsize=(9, 7))
    imagem = eixo.imshow(
        extracao.grade.score, origin="lower", cmap="magma", vmin=0, vmax=1, interpolation="nearest"
    )
    figura.colorbar(imagem, ax=eixo, label="fracao do vao piso-teto ocupada")
    eixo.set_title("Persistencia vertical por celula")
    figura.tight_layout()
    caminho = pasta / "02_persistencia.png"
    figura.savefig(caminho, dpi=130)
    plt.close(figura)
    gravadas.append(caminho)

    # 3. Mascara com as retas detectadas: o diagnostico mais util para ajustar.
    figura, eixo = plt.subplots(figsize=(9, 7))
    eixo.imshow(extracao.mascara, origin="lower", cmap="Greys", interpolation="nearest")
    res, origem = extracao.grade.res, extracao.grade.origem
    for parede in extracao.paredes:
        cantos = parede.contorno()
        pixels = (cantos - origem) / res
        fechado = np.vstack([pixels, pixels[:1]])
        eixo.plot(fechado[:, 0], fechado[:, 1], color="tab:red", lw=1.0)
    for vao in vaos:
        parede = next((p for p in extracao.paredes if p.nome == vao.parede), None)
        if parede is None:
            continue
        cor = {"porta": "tab:green", "passagem": "tab:green", "janela": "tab:blue"}.get(
            vao.classe, "tab:orange"
        )
        pontos = np.array(
            [
                parede.normal * parede.eixo_central + parede.direcao * vao.inicio,
                parede.normal * parede.eixo_central + parede.direcao * vao.fim,
            ]
        )
        pixels = (pontos - origem) / res
        eixo.plot(pixels[:, 0], pixels[:, 1], color=cor, lw=2.5)
    eixo.set_title("Mascara de parede, paredes vetorizadas (vermelho) e vaos")
    figura.tight_layout()
    caminho = pasta / "03_mascara_e_paredes.png"
    figura.savefig(caminho, dpi=140)
    plt.close(figura)
    gravadas.append(caminho)

    # 4. Previa da planta, so com a geometria vetorial que vai para o DXF.
    caminho = pasta / "04_previa_planta.png"
    previa_planta(extracao, vaos, caminho)
    gravadas.append(caminho)

    # 5. Mapas (s, h) das paredes com mais vaos: onde a etapa 4 decide.
    com_vaos = sorted(
        extracao.paredes,
        key=lambda p: -sum(1 for v in vaos if v.parede == p.nome),
    )[:max_paredes_sh]
    for parede in com_vaos:
        analise = analises.get(parede.nome)
        if analise is None or analise.material.size <= 1:
            continue
        figura, eixo = plt.subplots(figsize=(10, 3.2))
        extensao = [
            analise.s0,
            analise.s0 + analise.material.shape[0] * analise.celula,
            analise.h0,
            analise.h0 + analise.material.shape[1] * analise.celula,
        ]
        eixo.imshow(
            analise.material.T, origin="lower", extent=extensao, cmap="Greys",
            aspect="auto", interpolation="nearest",
        )
        for vao in analise.vaos:
            cor = {"porta": "tab:green", "passagem": "tab:green", "janela": "tab:blue"}.get(
                vao.classe, "tab:orange"
            )
            eixo.add_patch(
                plt.Rectangle(
                    (vao.inicio, vao.base), vao.largura, vao.altura,
                    fill=False, edgecolor=cor, lw=2.0,
                )
            )
            eixo.text(
                vao.centro, vao.topo + 0.05, vao.classe, color=cor,
                ha="center", fontsize=8,
            )
        eixo.set_xlabel("s ao longo da parede (m)")
        eixo.set_ylabel("altura (m)")
        eixo.set_title(f"{parede.nome}: alvenaria no plano da parede")
        figura.tight_layout()
        caminho = pasta / f"05_parede_{parede.nome}.png"
        figura.savefig(caminho, dpi=130)
        plt.close(figura)
        gravadas.append(caminho)

    return gravadas


def previa_planta(extracao: Extracao, vaos: list[Vao], caminho: str | Path) -> Path:
    """Desenha a planta vetorial em PNG, com a mesma geometria gravada no DXF."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from plantabaixa.dxf import trechos_de_alvenaria

    figura, eixo = plt.subplots(figsize=(11, 8))
    for parede in extracao.paredes:
        for s_a, s_b in trechos_de_alvenaria(parede, vaos, 1.40):
            meia = parede.espessura / 2
            base = parede.normal * parede.eixo_central
            cantos = np.array(
                [
                    base + parede.normal * meia + parede.direcao * s_a,
                    base + parede.normal * meia + parede.direcao * s_b,
                    base - parede.normal * meia + parede.direcao * s_b,
                    base - parede.normal * meia + parede.direcao * s_a,
                ]
            )
            eixo.fill(*np.vstack([cantos, cantos[:1]]).T, color="0.25", lw=0)

    por_nome = {p.nome: p for p in extracao.paredes}
    for vao in vaos:
        parede = por_nome.get(vao.parede)
        if parede is None:
            continue
        eixo_p = parede.normal * parede.eixo_central
        a = eixo_p + parede.direcao * vao.inicio
        b = eixo_p + parede.direcao * vao.fim
        if vao.classe in ("porta", "passagem"):
            dobradica = a - parede.normal * vao.lado * parede.espessura / 2
            ponta = dobradica + parede.normal * vao.lado * vao.largura
            eixo.plot(*np.array([dobradica, ponta]).T, color="tab:green", lw=1.4)
            angulos = np.linspace(0, np.pi / 2, 24)
            giro = np.array(
                [
                    dobradica
                    + vao.largura * (np.cos(t) * parede.normal * vao.lado + np.sin(t) * parede.direcao)
                    for t in angulos
                ]
            )
            eixo.plot(giro[:, 0], giro[:, 1], color="tab:green", lw=0.8, ls="--")
        elif vao.classe == "janela":
            for sinal in (1.0, -1.0):
                desloc = parede.normal * sinal * parede.espessura / 2
                eixo.plot(*np.array([a + desloc, b + desloc]).T, color="tab:blue", lw=1.6)
        else:
            eixo.plot(*np.array([a, b]).T, color="tab:orange", lw=2.5, alpha=0.7)

    eixo.set_aspect("equal")
    eixo.set_xlabel("x (m)")
    eixo.set_ylabel("y (m)")
    eixo.set_title("Previa da planta baixa: alvenaria, portas (verde), janelas (azul), oclusao (laranja)")
    eixo.grid(alpha=0.2, lw=0.4)
    figura.tight_layout()
    figura.savefig(caminho, dpi=140)
    plt.close(figura)
    return Path(caminho)
