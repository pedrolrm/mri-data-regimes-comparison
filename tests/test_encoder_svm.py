"""
Suíte de Testes Automatizados para a Integração ResNet-50 Encoder -> Classificador SVM.

Executa testes unitários e de integração utilizando o framework nativo `unittest`:
1. Arquitetura e integridade do extrator ResNet50Encoder (dimensão 2048, congelamento).
2. Treinamento, normalização, inferência e serialização do ClassificadorAlzheimerSVM.
3. Fluxo ponta-a-ponta integrado: Tensores MRI -> Espaço Latente -> Predição SVM.
4. Validação matemática das métricas clínicas de avaliação (Sensibilidade e Especificidade).

Execução via terminal:
    python tests/test_encoder_svm.py
    ou
    python -m unittest tests/test_encoder_svm.py
"""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.metrics import confusion_matrix, recall_score
import torch
import torch.nn as nn

# Garante a raiz do repositório no sys.path
_RAIZ_PROJETO = Path(__file__).resolve().parent.parent
if str(_RAIZ_PROJETO) not in sys.path:
    sys.path.insert(0, str(_RAIZ_PROJETO))

from src.models.encoder import ResNet50Encoder
from src.models.svm import ClassificadorAlzheimerSVM


class TestResNet50Encoder(unittest.TestCase):
    """Testes unitários para o extrator ResNet50Encoder."""

    def setUp(self) -> None:
        # Inicializa o encoder sem baixar os pesos remotos para execução rápida e determinística
        self.encoder = ResNet50Encoder(
            carregar_radimagenet=False,
            congelar_parametros=True,
            device="cpu"
        )

    def test_arquitetura_e_dimensao_saida(self) -> None:
        """Verifica se a ResNet-50 mapeia (Batch, 3, 224, 224) para (Batch, 2048)."""
        batch_size = 2
        lote_sintetico = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            saida = self.encoder(lote_sintetico)

        self.assertEqual(saida.shape, (batch_size, 2048))
        self.assertEqual(self.encoder.dim_saida, 2048)

    def test_fc_substituida_por_identity(self) -> None:
        """Assegura que a camada fc original de 1000 classes foi substituída por Identity."""
        self.assertIsInstance(self.encoder.model.fc, nn.Identity)

    def test_congelamento_e_descongelamento_pesos(self) -> None:
        """Verifica se as flags requires_grad são controladas adequadamente."""
        # Inicialmente todos devem estar congelados
        todos_congelados = all(not p.requires_grad for p in self.encoder.model.parameters())
        self.assertTrue(todos_congelados, "Os parâmetros do encoder deveriam estar congelados.")

        # Ao descongelar
        self.encoder.descongelar_pesos()
        todos_ativos = all(p.requires_grad for p in self.encoder.model.parameters())
        self.assertTrue(todos_ativos, "Os parâmetros deveriam estar ativos após descongelar_pesos.")

        # Ao congelar novamente
        self.encoder.congelar_pesos()
        todos_congelados_novamente = all(not p.requires_grad for p in self.encoder.model.parameters())
        self.assertTrue(todos_congelados_novamente, "Os parâmetros deveriam estar congelados novamente.")


class TestClassificadorAlzheimerSVM(unittest.TestCase):
    """Testes unitários para o wrapper ClassificadorAlzheimerSVM."""

    def setUp(self) -> None:
        np.random.seed(42)
        # Gera 60 amostras sintéticas de 2048 dimensões balanceadas em 2 classes (0 e 1)
        self.X_treino = np.random.randn(60, 2048)
        self.y_treino = np.array([0] * 30 + [1] * 30)

        self.X_teste = np.random.randn(20, 2048)
        self.y_teste = np.array([0] * 10 + [1] * 10)

        self.classificador = ClassificadorAlzheimerSVM(C=1.0, kernel="rbf", random_state=42)

    def test_erro_ao_prever_sem_treino(self) -> None:
        """Garante que NotFittedError é lançado se prever for chamado antes do treino."""
        with self.assertRaises(NotFittedError):
            self.classificador.prever(self.X_teste)

    def test_treino_e_predicao(self) -> None:
        """Testa o ciclo de ajuste e inferência com classes válidas."""
        self.assertFalse(self.classificador.esta_treinado)
        self.classificador.treinar(self.X_treino, self.y_treino)
        self.assertTrue(self.classificador.esta_treinado)

        predicoes = self.classificador.prever(self.X_teste)
        self.assertEqual(len(predicoes), len(self.X_teste))
        self.assertTrue(set(predicoes).issubset({0, 1}))

    def test_dimensoes_incompativeis(self) -> None:
        """Verifica a validação de formato das entradas X e y."""
        with self.assertRaises(ValueError):
            # X unidimensional
            self.classificador.treinar(np.array([1, 2, 3]), np.array([0, 1, 0]))

        with self.assertRaises(ValueError):
            # Número de amostras diferente entre X e y
            self.classificador.treinar(self.X_treino[:10], self.y_treino[:20])

    def test_serializacao_e_recuperacao_joblib(self) -> None:
        """Testa se salvar_modelo e carregar_modelo produzem predições 100% idênticas."""
        self.classificador.treinar(self.X_treino, self.y_treino)
        predicoes_originais = self.classificador.prever(self.X_teste)

        with tempfile.TemporaryDirectory() as dir_temp:
            caminho_arquivo = Path(dir_temp) / "svm_teste_serializacao.joblib"
            self.classificador.salvar_modelo(caminho_arquivo)
            self.assertTrue(caminho_arquivo.exists())

            # Carrega em nova instância
            modelo_restaurado = ClassificadorAlzheimerSVM.carregar_modelo(caminho_arquivo)
            self.assertTrue(modelo_restaurado.esta_treinado)

            predicoes_restauradas = modelo_restaurado.prever(self.X_teste)
            np.testing.assert_array_equal(
                predicoes_originais,
                predicoes_restauradas,
                err_msg="As predições após a desserialização divergiram do modelo original."
            )


class TestPipelineIntegradoEncoderSVM(unittest.TestCase):
    """Teste de integração do fluxo completo: Imagem -> Encoder -> Vetor 2048 -> SVM."""

    def test_fluxo_ponta_a_ponta(self) -> None:
        """Executa um pipeline de inferência completo sobre tensores de imagem sintéticos."""
        torch.manual_seed(42)
        np.random.seed(42)

        encoder = ResNet50Encoder(carregar_radimagenet=False, congelar_parametros=True)
        svm = ClassificadorAlzheimerSVM(C=1.0, kernel="rbf", random_state=42)

        # 1. Simula treino com dados extraídos
        imagens_treino = torch.randn(20, 3, 224, 224)
        rotulos_treino = np.array([0] * 10 + [1] * 10)

        with torch.no_grad():
            features_treino = encoder(imagens_treino).numpy()

        self.assertEqual(features_treino.shape, (20, 2048))
        svm.treinar(features_treino, rotulos_treino)

        # 2. Simula inferência em novas imagens
        imagens_teste = torch.randn(6, 3, 224, 224)
        with torch.no_grad():
            features_teste = encoder(imagens_teste).numpy()

        predicoes_finais = svm.prever(features_teste)

        self.assertEqual(len(predicoes_finais), 6)
        self.assertTrue(all(p in [0, 1] for p in predicoes_finais))


class TestMetricasClinicas(unittest.TestCase):
    """Validação da computação matemática de Sensibilidade e Especificidade."""

    def test_calculo_sensibilidade_e_especificidade(self) -> None:
        """
        Cenário com matriz de confusão pré-definida:
        TN = 80 (NonDemented classificados como NonDemented)
        FP = 20 (NonDemented classificados erroneamente como ModerateDemented)
        FN = 10 (ModerateDemented classificados erroneamente como NonDemented)
        TP = 90 (ModerateDemented classificados como ModerateDemented)
        """
        y_real = np.array([0] * 100 + [1] * 100)
        y_pred = np.array([0] * 80 + [1] * 20 + [0] * 10 + [1] * 90)

        tn, fp, fn, tp = confusion_matrix(y_real, y_pred).ravel()
        self.assertEqual((tn, fp, fn, tp), (80, 20, 10, 90))

        # Sensibilidade = TP / (TP + FN) = 90 / 100 = 0.90
        sensibilidade = recall_score(y_real, y_pred, pos_label=1)
        self.assertAlmostEqual(sensibilidade, 0.90, places=4)

        # Especificidade = TN / (TN + FP) = 80 / 100 = 0.80
        especificidade = tn / (tn + fp)
        self.assertAlmostEqual(especificidade, 0.80, places=4)


if __name__ == "__main__":
    print("=" * 70)
    print("EXECUTANDO SUÍTE DE TESTES: PIPELINE RESNET-50 ENCODER -> SVM")
    print("=" * 70)
    unittest.main(verbosity=2)
