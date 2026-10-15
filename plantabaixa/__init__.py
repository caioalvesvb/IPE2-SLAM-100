"""Extracao de planta baixa a partir de nuvem de pontos registrada do SLAM 100.

Pipeline em seis etapas, cada uma num modulo testavel isoladamente:

1. `nuvem`   - carga em blocos, voxelizacao e nivelamento por gravidade;
2. `paredes` - mascara de parede por persistencia vertical e vetorizacao;
3. `vaos`    - deteccao de portas e janelas por visada atraves do plano;
4. `dxf`     - escrita do DXF em camadas;
5. `qc`      - relatorio e figuras de diagnostico;
6. `planta_baixa.py` (raiz) - linha de comando que orquestra as etapas.

Uso programatico:

```python
from plantabaixa import gerar_planta
resultado = gerar_planta("../SN_000055_certo", saida="planta.dxf")
print(resultado.resumo())
```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from plantabaixa.erros import PlantaError

__all__ = ["PlantaError", "Resultado", "gerar_planta"]

if TYPE_CHECKING:  # pragma: no cover - apenas para ferramentas de tipagem
    from plantabaixa.pipeline import Resultado, gerar_planta


def __getattr__(nome: str) -> Any:
    """Importa `pipeline` so quando pedido.

    Assim `from plantabaixa import nuvem` nao arrasta laspy e ezdxf, o que
    mantem os testes de geometria leves e independentes de I/O.
    """
    if nome in ("Resultado", "gerar_planta"):
        from plantabaixa import pipeline

        return getattr(pipeline, nome)
    raise AttributeError(f"module {__name__!r} has no attribute {nome!r}")
