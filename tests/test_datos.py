"""El dataset cumple lo que pide el proyecto y la división es reproducible."""
from src.config import VARIABLES
from src.datos import cargar, dividir


def test_dataset_tiene_al_menos_1000_registros_y_19_variables():
    df = cargar()
    assert len(df) >= 1000
    assert len(VARIABLES) == 19


def test_no_quedan_vacios_despues_de_limpiar():
    assert cargar()[VARIABLES].isna().sum().sum() == 0


def test_division_reproducible_y_estratificada():
    a, b = dividir(cargar()), dividir(cargar())
    assert a[0].index.equals(b[0].index)                                  # misma semilla, mismos clientes
    assert abs(a[2].mean() - a[3].mean()) < 0.01                          # misma proporción de churn
