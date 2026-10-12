# Documentação Técnica: Treinamento Convolucional ResNet-50 com RadImageNet
**Projeto:** `mri-data-regimes-comparison`  
**Objetivo Científico:** Avaliação do impacto de regimes de volume de dados (*sample efficiency*) comparando Aprendizado de Máquina Clássico (SVM + Extrator) e Deep Learning Convolucional (ResNet-50) na classificação dos estágios da Doença de Alzheimer em imagens de Ressonância Magnética (MRI).  
**Data:** 08 de Outubro de 2026  
**Hardware Alvo:** NVIDIA GeForce RTX 5060 (8GB VRAM) — Arquitetura Blackwell  

---

## 1. Visão Geral e Contexto do Estudo

O estudo investiga o comportamento de diferentes paradigmas de Inteligência Artificial sob regimes amostrais restritos e abundantes:
* **Classes Alvo (4 estágios):** `MildDemented`, `ModerateDemented`, `NonDemented`, `VeryMildDemented`.
* **Dataset:** 40.000 imagens cerebrais de ressonância magnética (Kaggle Alzheimer MRI Dataset).
* **Divisão de Regimes Amostrais (80% Treino / 20% Teste estratificado):**
  1. **Regime 1:** $N = 1.000$ amostras (Treino: 800 | Teste: 200 — 50/classe)
  2. **Regime 2:** $N = 5.000$ amostras (Treino: 4.000 | Teste: 1.000 — 250/classe)
  3. **Regime 3:** $N = 10.000$ amostras (Treino: 8.000 | Teste: 2.000 — 500/classe)
  4. **Regime 4:** Full Dataset ($N = 40.000$ amostras — Treino: 32.000 | Teste: 8.000)
* **Pesos Pré-treinados Especializados:** `RadImageNet` (`Lab-Rasool/RadImageNet`), modelo pré-treinado em milhões de imagens radiológicas reais, provendo representações anatômicas superiores aos pesos genéricos do ImageNet tradicional.

---

## 2. Diagnóstico Crítico do Pipeline Herdado (SVM & Encoder)

Ao analisar a discrepância de performance reportada no classificador SVM inicial (~38% a 41% de acurácia, próximo ao chute aleatório de 25% para 4 classes), foi identificada a seguinte causa raiz:

### 2.1. Incompatibilidade Estrutural de Chaves no Checkpoint
* O arquivo oficial `ResNet50.pt` do RadImageNet armazena o backbone como um container sequencial (`nn.Sequential`):
  * `backbone.0.` $\rightarrow$ Primeira convolução (`conv1`)
  * `backbone.1.` $\rightarrow$ Batch Normalization inicial (`bn1`)
  * `backbone.4.` $\rightarrow$ Bloco residual 1 (`layer1`)
  * `backbone.5.` $\rightarrow$ Bloco residual 2 (`layer2`)
  * `backbone.6.` $\rightarrow$ Bloco residual 3 (`layer3`)
  * `backbone.7.` $\rightarrow$ Bloco residual 4 (`layer4`)
* O extrator original (`src/models/encoder.py`) aplicava apenas um corte de prefixo simples (`chave.replace("backbone.", "")`), resultando em chaves como `0.weight` e `4.0.conv1.weight`.
* Como a arquitetura padrão do PyTorch Torchvision espera nomes nominais (`conv1.weight`, `layer1.0.conv1.weight`) e a chamada de carregamento utilizava `strict=False`, o PyTorch **descartou silenciosamente todos os 318 tensores convolucionais**.
* **Consequência:** A rede operava com inicialização 100% pseudo-aleatória. O SVM estava classificando vetores sem aprendizado radiológico prévio.
* **Ação Tomada:** O arquivo do colaborador foi mantido em seu estado original por solicitação do usuário para que o parceiro possa aplicar a correção por conta própria. O mapeamento correto foi integrado à implementação da ResNet-50.

---

## 3. Implementação da Arquitetura Convolucional (`src/models/resnet50.py`)

Desenvolveu-se a classe `ResNet50Classifier` com os seguintes pilares técnicos:

1. **Mapeamento Exato de Pesos:**
   Implementação de dicionário de tradução que direciona as camadas sequenciais do RadImageNet para os atributos do Torchvision, assegurando o carregamento íntegro dos 318 tensores pré-treinados:
   ```python
   mapeamento_radimagenet = {
       "backbone.0.": "conv1.",
       "backbone.1.": "bn1.",
       "backbone.4.": "layer1.",
       "backbone.5.": "layer2.",
       "backbone.6.": "layer3.",
       "backbone.7.": "layer4.",
   }
   ```
2. **Nova Camada de Classificação:**
   Substituição da camada final original de 1.000 classes por:
   ```python
   self.model.fc = nn.Sequential(
       nn.Dropout(p=dropout_rate),  # Regularização contra overfitting em N reduzido
       nn.Linear(2048, num_classes) # num_classes = 4
   )
   ```
3. **Controle de Gradientes e Transfer Learning:**
   * `freeze_backbone()`: Congela parâmetros convolucionais para a fase de aquecimento (*warmup*).
   * `unfreeze_backbone()`: Libera a rede inteira para ajuste fino (*fine-tuning*).
   * `get_parameter_groups()`: Separa grupos de otimização permitindo taxas de aprendizado diferenciais (taxa menor para representações pré-treinadas e maior para a cabeça classificadora).
4. **Resolução de Autenticação Hugging Face:**
   Inclusão de `token=False` na chamada `hf_hub_download`, permitindo download anônimo público e suprimindo avisos espúrios de tokens expirados no ambiente do usuário.

---

## 4. Otimizações de Engenharia de Hardware para RTX 5060 (8GB)

O script `src/train_resnet.py` e o ponto de entrada `train.py` foram desenvolvidos especificamente para extrair a máxima eficiência computacional da arquitetura Blackwell:

| Técnica | Implementação no Código | Benefício Observado |
| :--- | :--- | :--- |
| **AMP com `bfloat16`** | `torch.amp.autocast(..., dtype=torch.bfloat16)` | Suporte nativo por hardware sem necessidade de escalonador dinâmico (`GradScaler`), ganho de ~2.5x em velocidade e redução de 50% em VRAM. |
| **TF32 em Tensor Cores** | `torch.set_float32_matmul_precision("high")` | Aceleração de multiplicações de matrizes com 19 bits de precisão sem perda de acurácia. |
| **cuDNN Benchmark** | `torch.backends.cudnn.benchmark = True` | Otimização estática dos algoritmos de convolução para entradas fixas de 224x224. |
| **DataLoaders Otimizados** | `pin_memory=True`, `prefetch_factor=2` | Transferência assíncrona da RAM para a memória da GPU com buffers pré-alocados. |
| **Estabilidade no Windows** | `num_workers=2` com `persistent_workers=True` | Elimina o custo de reinicialização de processos filhos via `multiprocessing.spawn`. |
| **Gerenciamento de VRAM** | `optimizer.zero_grad(set_to_none=True)` | Redução de largura de banda de memória por descarte de gradientes em vez de zeramento. |

---

## 5. Pipeline de Treinamento e Registro de Experimentos

### 5.1. Dinâmica de Treinamento em 2 Fases
* **Fase 1 (Épocas 1 a 2):** Backbone congelado. Apenas a camada linear recém-inicializada é ajustada, evitando gradientes ruidosos iniciais que degradam os pesos radiológicos.
* **Fase 2 (Época 3 em diante):** Descongelamento completo com otimizador `AdamW` e escalonador `CosineAnnealingLR` (taxas diferenciais: $1 \times 10^{-4}$ para o backbone e $1 \times 10^{-3}$ para o classificador).

### 5.2. Sistema de Logs e Métricas
* Classe `DualLogger` para registrar em tempo real tanto no console quanto em arquivo estruturado `.txt` (`outputs/logs/treinamento_resnet50_<regime>_<timestamp>.txt`).
* Métricas computadas por época: Perda de treino, acurácia de treino, perda de validação, acurácia de validação, Macro F1-Score, tempo de ciclo e alocação máxima de memória VRAM (MB).
* Avaliação final utilizando o melhor ponto de verificação (*checkpoint* salvo por Early Stopping):
  * Relatório de Classificação (*Precision*, *Recall*, *F1-Score*, *Support* por classe).
  * Matriz de Confusão formatada no padrão exato utilizado nas análises do artigo.

---

## 6. Diagnóstico e Resolução de Problemas no Ambiente

1. **Execução no ambiente `venv` (CPU-only):**
   * **Problema:** O usuário ativou o ambiente `venv` local, o qual possuía o pacote `torch 2.14.1+cpu`. Isso desativou a detecção de CUDA e disparou avisos de execução lenta em CPU.
   * **Solução:** Foi demonstrado que o ambiente raiz do Conda (`miniconda3`) já possuía a versão otimizada com suporte à RTX 5060 (`torch 2.12.0.dev+cu128` com CUDA 12.8). A instrução fornecida foi rodar `deactivate` no PowerShell ou atualizar o `venv` com o repositório nightly cu128.
2. **Proteção de Criação de Diretórios:**
   * Garantida a chamada recursiva de `mkdir(parents=True, exist_ok=True)` antes de qualquer serialização com `torch.save()`, evitando falhas caso a pasta `outputs/checkpoints` seja limpa durante execuções concorrentes.

---

## 7. Resultados Preliminares de Validação de Hardware

Execução de verificação de sanidade computacional na NVIDIA GeForce RTX 5060:
* **Tempo de Época ($N=1.000$):** ~5.3 segundos.
* **Consumo de Memória de Vídeo:** ~722 MB (de um total de 8.192 MB disponíveis).
* **Desempenho Preliminar (Época 1 - Fase Linear Probing):**
  * Acurácia: **55.00%**
  * Macro F1-Score: **0.4935**
  * *(Superior ao resultado de ~38% a 41% obtido com o extrator não inicializado do SVM).*

---

## 8. Guia de Reprodução e Execução

### Treinar um Regime Específico
```powershell
# Regime 1 (N = 1.000)
python train.py --regime 1000 --epochs 20 --batch_size 64

# Regime 2 (N = 5.000)
python train.py --regime 5000 --epochs 20 --batch_size 64

# Regime 3 (N = 10.000)
python train.py --regime 10000 --epochs 20 --batch_size 64

# Regime 4 (Dataset Completo - 40.000)
python train.py --epochs 25 --batch_size 64
```

### Executar Comparativo Completo para o Artigo
```powershell
python train.py --all_regimes --epochs 20 --batch_size 64
```
*Gera uma tabela consolidada ao término de todos os 4 regimes, exportando o log final para a pasta `outputs/logs/`.*

---

## 9. Extensão: Classificação Binária (`train_binary.py`)

Para complementar o artigo com análises diagnósticas complementares, foi desenvolvido o módulo `src/train_binary.py` (e atalho raiz `train_binary.py`), permitindo duas formulações binárias com a ResNet-50 (`num_classes=2`):

### 9.1. Formulação 1: Extremos Clínicos (`--task moderate`)
* **Classes:** `NonDemented` (0) vs. `ModerateDemented` (1).
* **Amostras:** 20.000 imagens ($10.000$ saudáveis vs. $10.000$ demência moderada — 50/50 balanceado).
* **Objetivo:** Isolar o contraste anatômico máximo (cérebro totalmente íntegro vs. atrofia cortical severa consolidada), descartando ruído de transição dos estágios limítrofes (*Mild* e *VeryMild*).

### 9.2. Formulação 2: Diagnóstico Global Agrupado (`--task grouped`)
* **Classes:** `NonDemented` (0) vs. `Demented` (1, agrupando `MildDemented` + `ModerateDemented` + `VeryMildDemented`).
* **Amostras:**
  * **Modo Balanceado (Padrão):** $10.000$ `NonDemented` vs. $10.000$ `Demented` (amostragem equilibrada de ~3.334 de cada subestágio demente).
  * **Modo Completo (`--no_balance`):** $10.000$ `NonDemented` vs. $30.000$ `Demented` (utiliza as 40k com ponderação de perda $[3.0, 1.0]$ na *CrossEntropyLoss*).
* **Objetivo:** Avaliar a triagem clínica binária global (*Demente* vs. *Saudável*), onde qualquer grau de demência deve ser detectado.

### 9.3. Comandos de Execução
```powershell
# Rodar todos os regimes (1k, 5k, 10k, 20k) no experimento agrupado:
python train_binary.py --task grouped --all_regimes --epochs 10 --batch_size 64 --seed 42

# Rodar todos os regimes (1k, 5k, 10k, 20k) em ambos os experimentos binários:
python train_binary.py --task both --all_regimes --epochs 10 --batch_size 64 --seed 42
```

### 9.4. Resultados Empíricos dos Regimes Binários

#### Experimento Agrupado (NonDemented vs. Demented - 1:1)
| Regime ($N$) | $N_{\text{Treino}}$ | Tempo (s) | Acurácia | Sensibilidade | Especificidade | F1-Score |
|---|---|---|---|---|---|---|
| $1.000$ | $800$ | 35.3 s | 68.50% | 62.00% | 75.00% | 0.6837 |
| $5.000$ | $4.000$ | 109.3 s | 76.70% | 70.80% | 82.60% | 0.7662 |
| $10.000$ | $8.000$ | 200.5 s | 78.85% | 74.50% | 83.20% | 0.7881 |
| $20.000$ | $16.000$ | 386.5 s | 81.88% | 80.80% | 82.95% | 0.8187 |

#### Experimento Extremos (NonDemented vs. ModerateDemented)
| Regime ($N$) | $N_{\text{Treino}}$ | Tempo (s) | Acurácia | Sensibilidade | Especificidade | F1-Score |
|---|---|---|---|---|---|---|
| $1.000$ | $800$ | 76.6 s | 84.00% | 71.00% | 97.00% | 0.8372 |
| $5.000$ | $4.000$ | 124.1 s | 97.40% | 96.80% | 98.00% | 0.9740 |
| $10.000$ | $8.000$ | 217.6 s | 98.95% | 99.00% | 98.90% | 0.9895 |
| $20.000$ | $16.000$ | 421.1 s | 99.78% | 99.75% | 99.80% | 0.9977 |


