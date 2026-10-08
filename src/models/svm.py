"""
Módulo do Classificador Support Vector Machine (SVM) para Estágios da Doença de Alzheimer.

Este módulo define a classe `ClassificadorAlzheimerSVM`, um wrapper modular e orientado
a objetos construído sobre o ecossistema scikit-learn. O pipeline do projeto integra
uma ResNet-50 pré-treinada no RadImageNet como extrator estático de características
(produzindo vetores densos de 2048 dimensões) e este classificador SVM para inferência
dos estágios de Alzheimer.

Dado que os algoritmos de Support Vector Machine são altamente sensíveis à escala das
variáveis de entrada (devido ao cálculo de distâncias euclidianas na definição dos hiperplanos
de margem máxima), esta classe encapsula o pré-processamento via `StandardScaler` em conjunto
com o estimador `SVC`, garantindo que não ocorra vazamento de dados (data leakage) entre
os conjuntos de treino e teste.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import joblib
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class _MetodoCarregadorHibrido:
    """
    Descritor que viabiliza a chamada de `carregar_modelo` tanto a partir de uma instância
    existente (atualizando seu estado interno in-place) quanto diretamente pela classe
    (atuando como factory method e retornando uma nova instância inicializada).
    """

    def __init__(self, metodo_instancia: Any) -> None:
        self.metodo_instancia = metodo_instancia

    def __get__(self, instance: Any, owner: Optional[type] = None) -> Any:
        if instance is None:
            def _chamada_classe(caminho: Union[str, Path]) -> Any:
                nova_instancia = owner()  # type: ignore[misc]
                return nova_instancia.carregar_modelo(caminho)

            _chamada_classe.__doc__ = self.metodo_instancia.__doc__
            return _chamada_classe
        return self.metodo_instancia.__get__(instance, owner)


class ClassificadorAlzheimerSVM:
    """
    Wrapper orientado a objetos para o classificador Support Vector Machine (SVM)
    integrado com normalização de características (StandardScaler).

    Encapsula o pipeline de escalonamento e classificação, permitindo fácil treino,
    inferência e serialização dos pesos e do estado do modelo.

    Atributos:
        scaler (StandardScaler): Instância do normalizador z-score do scikit-learn.
        modelo (SVC): Estimador Support Vector Classifier do scikit-learn.
        _esta_treinado (bool): Indicador do estado de ajuste do pipeline.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: Union[str, float] = "scale",
        degree: int = 3,
        coef0: float = 0.0,
        shrinking: bool = True,
        probability: bool = False,
        tol: float = 1e-3,
        cache_size: float = 200.0,
        class_weight: Optional[Union[str, Dict[Any, float]]] = None,
        max_iter: int = -1,
        decision_function_shape: str = "ovr",
        break_ties: bool = False,
        random_state: Optional[int] = 42,
        **kwargs: Any,
    ) -> None:
        """
        Inicializa o classificador SVM e o padronizador de dados.

        Parâmetros:
            C (float): Parâmetro de regularização do SVM. Valores menores aumentam a margem
                à custa de mais erros de classificação nos dados de treino. Padrão: 1.0.
            kernel (str): Função de núcleo utilizada no algoritmo ('linear', 'poly',
                'rbf', 'sigmoid', 'precomputed'). Padrão: 'rbf'.
            gamma (str ou float): Coeficiente do kernel para 'rbf', 'poly' e 'sigmoid'.
                Se 'scale', utiliza 1 / (n_features * X.var()). Padrão: 'scale'.
            degree (int): Grau da função de kernel polinomial ('poly'). Padrão: 3.
            coef0 (float): Termo independente em funções de kernel 'poly' e 'sigmoid'. Padrão: 0.0.
            shrinking (bool): Se True, utiliza a heurística de shrinking do algoritmo SMO. Padrão: True.
            probability (bool): Se True, ativa o cálculo de probabilidades calibradas via
                Platt scaling (aumenta o tempo de treino). Padrão: False.
            tol (float): Tolerância para o critério de parada. Padrão: 1e-3.
            cache_size (float): Tamanho do cache de kernel em MB. Padrão: 200.0.
            class_weight (str ou dict, opcional): Pesos associados às classes. Se 'balanced',
                ajusta pesos inversamente proporcionais às frequências de classe. Padrão: None.
            max_iter (int): Limite máximo de iterações no solucionador (-1 para ilimitado). Padrão: -1.
            decision_function_shape (str): Estratégia multiclasse ('ovr' para one-vs-rest,
                'ovo' para one-vs-one). Padrão: 'ovr'.
            break_ties (bool): Se True e decision_function_shape='ovr', desempata predições
                de acordo com os valores de confiança da função de decisão. Padrão: False.
            random_state (int, opcional): Semente para reprodutibilidade no cálculo de probabilidades. Padrão: 42.
            **kwargs (Any): Argumentos adicionais repassados diretamente ao construtor do `SVC`.
        """
        # Instanciação do StandardScaler para normalização de features sensíveis à escala
        self.scaler: StandardScaler = StandardScaler()

        # Instanciação do Support Vector Classifier com hiperparâmetros configuráveis
        self.modelo: SVC = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,
            **kwargs,
        )

        self._esta_treinado: bool = False

    @property
    def esta_treinado(self) -> bool:
        """
        Retorna se o pipeline (scaler e modelo SVC) já foi treinado com dados.

        Retorno:
            bool: True se treinado, False caso contrário.
        """
        return self._esta_treinado

    def treinar(
        self,
        X: Union[np.ndarray, list, Any],
        y: Union[np.ndarray, list, Any],
    ) -> "ClassificadorAlzheimerSVM":
        """
        Ajusta o normalizador (StandardScaler) aos dados de treino, transforma as
        características e treina o classificador SVM (SVC).

        Parâmetros:
            X (array-like de formato (n_amostras, n_features)): Matriz de entrada com os
                vetores de características extraídos pelo encoder (ex: 2048 dimensões).
            y (array-like de formato (n_amostras,)): Rótulos das classes alvo correspondentes
                aos estágios de Alzheimer.

        Retorno:
            ClassificadorAlzheimerSVM: A própria instância treinada (permite encadeamento de métodos).

        Lança:
            ValueError: Se os dados de entrada forem inválidos ou tiverem dimensões incompatíveis.
        """
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        if X_arr.ndim != 2:
            raise ValueError(
                f"A matriz de características X deve ser bidimensional (n_amostras, n_features). "
                f"Formato recebido: {X_arr.shape}."
            )

        if len(X_arr) != len(y_arr):
            raise ValueError(
                f"O número de amostras em X ({len(X_arr)}) difere do número de rótulos em y ({len(y_arr)})."
            )

        # 1. Ajusta o scaler e transforma os dados de treinamento
        X_escalado = self.scaler.fit_transform(X_arr)

        # 2. Treina o estimador SVC nos dados padronizados
        self.modelo.fit(X_escalado, y_arr)

        # 3. Atualiza o estado interno
        self._esta_treinado = True

        return self

    def prever(self, X: Union[np.ndarray, list, Any]) -> np.ndarray:
        """
        Aplica a padronização previamente calculada aos novos dados e infere as classes
        utilizando o modelo SVM treinado.

        Parâmetros:
            X (array-like de formato (n_amostras, n_features)): Matriz com novas amostras
                a serem classificadas.

        Retorno:
            np.ndarray: Vetor unidimensional com as classes preditas para cada amostra.

        Lança:
            NotFittedError: Se o classificador for chamado antes da execução de `treinar`.
            ValueError: Se o formato de X for inconsistente.
        """
        self._verificar_se_treinado()

        X_arr = np.asarray(X)
        if X_arr.ndim != 2:
            raise ValueError(
                f"A matriz de características X deve ser bidimensional (n_amostras, n_features). "
                f"Formato recebido: {X_arr.shape}."
            )

        # 1. Transforma as novas amostras utilizando a média e desvio padrão do treino
        X_escalado = self.scaler.transform(X_arr)

        # 2. Realiza a predição com o SVC
        return self.modelo.predict(X_escalado)

    def prever_probabilidades(self, X: Union[np.ndarray, list, Any]) -> np.ndarray:
        """
        Calcula as probabilidades estimadas para cada classe em novas amostras.

        Requer que o classificador tenha sido instanciado com `probability=True`.

        Parâmetros:
            X (array-like de formato (n_amostras, n_features)): Matriz com novas amostras.

        Retorno:
            np.ndarray: Matriz de formato (n_amostras, n_classes) contendo as probabilidades preditas.

        Lança:
            NotFittedError: Se o modelo ainda não tiver sido treinado.
            AttributeError: Se o modelo foi instanciado com `probability=False`.
        """
        self._verificar_se_treinado()

        if not getattr(self.modelo, "probability", False):
            raise AttributeError(
                "O cálculo de probabilidades requer que a classe tenha sido instanciada com 'probability=True'."
            )

        X_arr = np.asarray(X)
        X_escalado = self.scaler.transform(X_arr)
        return self.modelo.predict_proba(X_escalado)

    def salvar_modelo(self, caminho: Union[str, Path]) -> None:
        """
        Persiste o normalizador (StandardScaler) e o classificador (SVC) em um único
        arquivo utilizando a biblioteca joblib.

        Parâmetros:
            caminho (str ou Path): Caminho do arquivo onde o modelo serializado será salvo.

        Lança:
            NotFittedError: Se tentar salvar um modelo que ainda não foi treinado.
        """
        self._verificar_se_treinado()

        caminho_arquivo = Path(caminho)
        # Garante a criação dos diretórios pais, caso não existam
        caminho_arquivo.parent.mkdir(parents=True, exist_ok=True)

        estado = {
            "scaler": self.scaler,
            "modelo": self.modelo,
            "esta_treinado": self._esta_treinado,
        }

        joblib.dump(estado, caminho_arquivo)

    @_MetodoCarregadorHibrido
    def carregar_modelo(self, caminho: Union[str, Path]) -> "ClassificadorAlzheimerSVM":
        """
        Carrega o normalizador (StandardScaler) e o classificador (SVC) a partir de um
        arquivo serializado previamente com joblib.

        Este método pode ser executado a partir de uma instância existente (atualizando-a)
        ou diretamente a partir da classe `ClassificadorAlzheimerSVM.carregar_modelo(...)`,
        retornando uma nova instância pronta para uso.

        Parâmetros:
            caminho (str ou Path): Caminho para o arquivo salvo.

        Retorno:
            ClassificadorAlzheimerSVM: A instância com os pesos e scalers carregados.

        Lança:
            FileNotFoundError: Se o arquivo especificado não existir.
            KeyError: Se a estrutura do arquivo salvo for inválida ou incompatível.
        """
        caminho_arquivo = Path(caminho)
        if not caminho_arquivo.is_file():
            raise FileNotFoundError(f"Arquivo de modelo não encontrado: {caminho_arquivo}")

        conteudo = joblib.load(caminho_arquivo)

        if not isinstance(conteudo, dict) or "scaler" not in conteudo or "modelo" not in conteudo:
            raise KeyError(
                "O arquivo carregado não contém a estrutura esperada com 'scaler' e 'modelo'."
            )

        self.scaler = conteudo["scaler"]
        self.modelo = conteudo["modelo"]
        self._esta_treinado = conteudo.get("esta_treinado", True)

        return self

    @classmethod
    def carregar_de_arquivo(cls, caminho: Union[str, Path]) -> "ClassificadorAlzheimerSVM":
        """
        Método de fábrica (factory method) para instanciar e carregar diretamente um
        classificador salvo em disco.

        Parâmetros:
            caminho (str ou Path): Caminho para o arquivo salvo.

        Retorno:
            ClassificadorAlzheimerSVM: Nova instância configurada com o scaler e modelo recuperados.
        """
        instancia = cls()
        instancia.carregar_modelo(caminho)
        return instancia

    def _verificar_se_treinado(self) -> None:
        """
        Verifica se o classificador já foi ajustado. Caso contrário, lança NotFittedError.

        Lança:
            NotFittedError: Se o modelo ainda não tiver sido treinado.
        """
        if not self._esta_treinado:
            raise NotFittedError(
                "Esta instância de 'ClassificadorAlzheimerSVM' ainda não foi treinada. "
                "Chame o método 'treinar(X, y)' antes de realizar predições ou salvar o modelo."
            )


if __name__ == "__main__":
    # Teste de verificação rápida com dados sintéticos de 2048 dimensões (simulando ResNet-50)
    print("--- Teste do ClassificadorAlzheimerSVM ---")
    np.random.seed(42)

    # 100 amostras sintéticas com 2048 dimensões (mesma dimensão de saída da ResNet-50)
    X_sintetico = np.random.randn(100, 2048) * 15.0 + 5.0
    y_sintetico = np.random.choice([0, 1, 2, 3], size=100)  # 4 estágios de Alzheimer

    classificador = ClassificadorAlzheimerSVM(C=1.0, kernel="rbf", probability=True)
    classificador.treinar(X_sintetico, y_sintetico)

    predicoes = classificador.prever(X_sintetico[:5])
    print(f"Predições (primeiras 5 amostras): {predicoes}")

    caminho_teste = Path("temp_svm_teste.joblib")
    classificador.salvar_modelo(caminho_teste)
    print(f"Modelo salvo em: {caminho_teste}")

    # Teste 1: Carregamento direto pela classe (ClassificadorAlzheimerSVM.carregar_modelo)
    novo_classificador = ClassificadorAlzheimerSVM.carregar_modelo(caminho_teste)
    novas_predicoes = novo_classificador.prever(X_sintetico[:5])
    print(f"Predições (carregado via classe): {novas_predicoes}")
    assert np.array_equal(predicoes, novas_predicoes), "As predições divergiram no carregamento via classe!"

    # Teste 2: Carregamento através de instância existente
    instancia_existente = ClassificadorAlzheimerSVM()
    instancia_existente.carregar_modelo(caminho_teste)
    predicoes_instancia = instancia_existente.prever(X_sintetico[:5])
    print(f"Predições (carregado via instância): {predicoes_instancia}")
    assert np.array_equal(predicoes, predicoes_instancia), "As predições divergiram no carregamento via instância!"

    caminho_teste.unlink(missing_ok=True)
    print("Todos os testes foram concluídos com sucesso!")
