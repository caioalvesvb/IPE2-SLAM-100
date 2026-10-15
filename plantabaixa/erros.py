"""Excecao unica do pacote, no mesmo espirito do `FormatError` do converter.py."""

from __future__ import annotations


class PlantaError(ValueError):
    """Dados ausentes, inconsistentes ou insuficientes para gerar a planta.

    Erros de uso (arquivo inexistente, parametro invalido, nuvem sem piso
    identificavel) sobem como esta excecao para que a CLI os apresente como
    mensagem objetiva em vez de traceback.
    """
