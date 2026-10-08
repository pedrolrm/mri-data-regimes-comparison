"""
Script de Pré-processamento de Imagens de Ressonância Magnética (MRI).

Objetivo:
    Padronizar imagens de MRI cerebral para a dimensão 224x224 com redimensionamento
    proporcional (preservando o aspect ratio) e preenchimento simétrico (padding dinâmico
    com a cor exata do fundo da imagem), mantendo a anatomia cerebral e os biomarcadores
    de Alzheimer sem distorções anisotrópicas ou perda de córtex cerebral por cortes (CenterCrop).
"""

import argparse
from pathlib import Path
import sys
from PIL import Image


def obter_raiz_projeto() -> Path:
    """
    Identifica de forma robusta a raiz do repositório/projeto, independentemente
    de onde o comando ou módulo seja executado no terminal.
    """
    diretorio_script = Path(__file__).resolve().parent

    # Como o script reside dentro de 'utils/', a raiz padrão é o diretório pai
    candidato_raiz = diretorio_script.parent

    # Verifica se o candidato possui marcadores típicos do projeto
    if (candidato_raiz / ".git").exists() or (candidato_raiz / "data").exists():
        return candidato_raiz

    # Fallback procurando marcadores em ancestrais
    for ancestral in diretorio_script.parents:
        if (ancestral / ".git").exists() or (ancestral / "data").exists():
            return ancestral

    return candidato_raiz


def resolver_caminho(caminho: str | Path, raiz_projeto: Path) -> Path:
    """
    Resolve um caminho garantindo que caminhos relativos funcionem corretamente,
    seja a partir do diretório de trabalho atual (CWD) ou relativo à raiz do projeto.
    """
    p = Path(caminho)
    if p.is_absolute():
        return p

    # Se o caminho existir relativo ao diretório atual de execução do terminal
    if (Path.cwd() / p).exists():
        return (Path.cwd() / p).resolve()

    # Caso contrário, resolve em relação à raiz do projeto
    return (raiz_projeto / p).resolve()


def redimensionar_com_padding_simetrico(
    imagem: Image.Image,
    tamanho_alvo: int = 224
) -> Image.Image:
    """
    Redimensiona uma imagem mantendo rigorosamente a proporção original (aspect ratio)
    e aplica preenchimento (padding) simétrico com a cor real do fundo da imagem
    até que a imagem atinja exatamente a dimensão (tamanho_alvo x tamanho_alvo).

    Parâmetros:
        imagem (PIL.Image.Image): Imagem original carregada pelo Pillow.
        tamanho_alvo (int): Dimensão final desejada para largura e altura (padrão: 224).

    Retorna:
        PIL.Image.Image: Imagem resultante com dimensões exatas de tamanho_alvo x tamanho_alvo.
    """
    largura_orig, altura_orig = imagem.size

    # 1. Descobrir a maior dimensão da imagem original e calcular o fator de escala
    maior_dimensao = max(largura_orig, altura_orig)
    fator_escala = tamanho_alvo / maior_dimensao

    # Novas dimensões mantendo o aspect ratio
    nova_largura = int(round(largura_orig * fator_escala))
    nova_altura = int(round(altura_orig * fator_escala))

    metodo_resample = getattr(Image, "Resampling", Image).LANCZOS

    # Redimensionamento proporcional de alta qualidade
    imagem_redimensionada = imagem.resize(
        (nova_largura, nova_altura),
        resample=metodo_resample
    )

    # 2. DETECÇÃO DINÂMICA DO FUNDO
    # Lê a cor exata do pixel no canto superior esquerdo (x=0, y=0)
    # Como as imagens sofreram skull-stripping, este pixel é sempre o fundo.
    cor_fundo = imagem.getpixel((0, 0))

    # Cria a nova imagem de base usando a cor exata detectada
    modo = imagem.mode
    imagem_com_padding = Image.new(modo, (tamanho_alvo, tamanho_alvo), color=cor_fundo)

    # 3. Calcular o deslocamento para centralizar a imagem (padding simétrico)
    posicao_x = (tamanho_alvo - nova_largura) // 2
    posicao_y = (tamanho_alvo - nova_altura) // 2

    # Colar a imagem redimensionada no centro da imagem base
    imagem_com_padding.paste(imagem_redimensionada, (posicao_x, posicao_y))

    return imagem_com_padding


def processar_dataset(
    diretorio_origem: Path,
    diretorio_destino: Path,
    tamanho_alvo: int = 224,
    extensoes_suportadas: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
) -> None:
    """
    Percorre recursivamente a estrutura de pastas do diretório de origem, aplica o
    redimensionamento e padding simétrico em cada imagem, e salva no diretório de destino
    replicando a estrutura de classes e padronizando para 3 canais (RGB).

    Parâmetros:
        diretorio_origem (Path): Caminho da pasta raiz com as imagens originais.
        diretorio_destino (Path): Caminho da pasta onde as imagens tratadas serão salvas.
        tamanho_alvo (int): Dimensão final das imagens (padrão: 224).
        extensoes_suportadas (tuple): Extensões de arquivos de imagem a serem processadas.
    """
    if not diretorio_origem.exists():
        print(f"[ERRO] Diretório de origem não encontrado: {diretorio_origem}", file=sys.stderr)
        return

    # Se houver uma subpasta 'Data' contendo as classes, detecta automaticamente
    pasta_base_busca = diretorio_origem
    if (diretorio_origem / "Data").is_dir() and not any(
        p.is_dir() for p in diretorio_origem.iterdir() if p.name != "Data"
    ):
        pasta_base_busca = diretorio_origem / "Data"
        print(f"[INFO] Subpasta 'Data' detectada como raiz das classes: {pasta_base_busca}")

    # Localizar todas as imagens mantendo a estrutura relativa
    arquivos_imagem = [
        arquivo for arquivo in pasta_base_busca.rglob("*")
        if arquivo.is_file() and arquivo.suffix.lower() in extensoes_suportadas
    ]

    total_imagens = len(arquivos_imagem)
    if total_imagens == 0:
        print(f"[AVISO] Nenhuma imagem encontrada em: {pasta_base_busca}")
        return

    print("=" * 70)
    print("INICIANDO PRÉ-PROCESSAMENTO DO DATASET MRI")
    print(f"Diretório de Origem:  {pasta_base_busca.resolve()}")
    print(f"Diretório de Destino: {diretorio_destino.resolve()}")
    print(f"Tamanho Alvo:         {tamanho_alvo}x{tamanho_alvo}")
    print(f"Total de Imagens:     {total_imagens}")
    print("=" * 70)

    processadas_com_sucesso = 0
    erros = 0

    for indice, caminho_img in enumerate(arquivos_imagem, start=1):
        try:
            # Manter a estrutura de subpastas da classe
            caminho_relativo = caminho_img.relative_to(pasta_base_busca)
            destino_arquivo = diretorio_destino / caminho_relativo

            # Cria a pasta de destino correspondente se não existir
            destino_arquivo.parent.mkdir(parents=True, exist_ok=True)

            # Abrir e processar a imagem
            with Image.open(caminho_img) as img:
                formato_original = img.format  # Preserva o formato detectado (ex: JPEG, PNG)

                # Força a conversão para RGB (3 canais) para padronização em redes neurais
                img = img.convert("RGB")

                img_processada = redimensionar_com_padding_simetrico(img, tamanho_alvo=tamanho_alvo)

                # Salvar mantendo a extensão e o formato original do dataset com alta qualidade
                if formato_original == "JPEG" or caminho_img.suffix.lower() in (".jpg", ".jpeg"):
                    img_processada.save(destino_arquivo, format="JPEG", quality=95)
                else:
                    img_processada.save(destino_arquivo, format=formato_original)

            processadas_com_sucesso += 1

            # Exibição periódica do progresso
            if indice % 500 == 0 or indice == total_imagens:
                porcentagem = (indice / total_imagens) * 100
                print(f"Progresso: {indice}/{total_imagens} ({porcentagem:.1f}%) processadas...")

        except Exception as e:
            erros += 1
            print(f"[FALHA] Erro ao processar '{caminho_img.name}': {e}", file=sys.stderr)

    print("=" * 70)
    print("PRÉ-PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
    print(f"Imagens processadas com sucesso: {processadas_com_sucesso}/{total_imagens}")
    if erros > 0:
        print(f"Imagens com falhas: {erros}")
    print(f"Arquivos salvos em: {diretorio_destino.resolve()}")
    print("=" * 70)


def main():
    raiz_projeto = obter_raiz_projeto()
    origem_padrao = raiz_projeto / "data" / "raw"
    destino_padrao = raiz_projeto / "data" / "processed"

    parser = argparse.ArgumentParser(
        description="Pré-processamento de imagens MRI com redimensionamento proporcional e padding simétrico para 224x224."
    )
    parser.add_argument(
        "--origem",
        type=str,
        default=str(origem_padrao),
        help=f"Caminho do diretório de origem das imagens (padrão: {origem_padrao})"
    )
    parser.add_argument(
        "--destino",
        type=str,
        default=str(destino_padrao),
        help=f"Caminho do diretório de destino para salvar as imagens processadas (padrão: {destino_padrao})"
    )
    parser.add_argument(
        "--tamanho",
        type=int,
        default=224,
        help="Dimensão final desejada (largura e altura iguais, padrão: 224)"
    )

    args = parser.parse_args()

    diretorio_origem = resolver_caminho(args.origem, raiz_projeto)
    diretorio_destino = resolver_caminho(args.destino, raiz_projeto)

    processar_dataset(
        diretorio_origem=diretorio_origem,
        diretorio_destino=diretorio_destino,
        tamanho_alvo=args.tamanho
    )


if __name__ == "__main__":
    main()
