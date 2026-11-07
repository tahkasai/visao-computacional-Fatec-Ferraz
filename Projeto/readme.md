# Sistema Integrado de Processamento Digital de Imagens e Vídeo (PDI)

Projeto acadêmico final desenvolvido em Python e OpenCV para a disciplina de **Processamento Digital de Imagens / Visão Computacional**, estruturado em módulos independentes e acessível via menu interativo de terminal e modo de demonstração automatizado.

---

## 1. Visão Geral da Arquitetura

O sistema foi concebido com rigor modular e orientação a objetos, dividindo as operações em classes especializadas para facilitar a manutenção, teste unitário e extensão:

| Módulo / Classe | Responsabilidade Técnica | Funcionalidades Implementadas |
| :--- | :--- | :--- |
| `ImageIOManager` | Entrada, saída e exibição | Leitura (`cv2.imread`), cópia profunda (`copy`), exibição simultânea lado a lado (Original vs. Processada com Matplotlib) e salvamento (`cv2.imwrite`). |
| `BasicOperations` | Transformações pontuais e espaciais | Inversão (negativo), ajuste linear de brilho e contraste, adição/fusão ponderada, subtração, conversão cinza, conversão BGR $\leftrightarrow$ HSV, redimensionamento e recorte de ROI. |
| `FilteringAndEnhancement` | Filtragem espacial e detecção de bordas | Filtro da média (Box filter), mediana (não-linear contra ruído impulsivo), gaussiano 2D, operador diferencial de Sobel (magnitude combinada) e detector ótimo de Canny. |
| `HistogramAndIntensity` | Domínio de intensidades | Gráfico de distribuição de frequências (cinza e 3 canais BGR), equalização de histograma (equalização do canal V no HSV para imagens coloridas), splitting de canais B, G, R, binarização (limiar fixo, adaptativo gaussiano e método de Otsu) e quantização (cinza uniforme e agrupamento K-Means de cores). |
| `SegmentationAndMorphology` | Segmentação e morfologia matemática | Algoritmo de Crescimento de Regiões (*Region Growing* baseado em fila e conectividade 8 conforme Gonzalez & Woods) e morfologia estrutural (erosão, dilatação, abertura e fechamento com elemento retangular, elíptico ou em cruz). |
| `FaceRecognitionSystem` | Reconhecimento facial (Antonello, 2017) | Detecção por classificadores em cascata de Haar (`haarcascade_frontalface_default.xml`) e reconhecimento por Local Binary Pattern Histograms (LBPH), integrando `cv2.face.LBPHFaceRecognizer_create()` com fallback nativo em NumPy. |
| `AutomatedAttendanceSystem` | Desafio adicional: Chamada inteligente | Detecção de múltiplos alunos em fotos de sala de aula ou vídeo em tempo real (webcam), identificação automática por face com índice de confiança e geração de ata em arquivo CSV (`chamada_presenca.csv`). |
| `PDIFinalProjectApp` | Controlador e menu interativo | Interface de linha de comando estruturada em submenus com tratamento de exceções, modo interativo e rotina de demonstração automatizada (`--demo`). |

---

## 2. Configuração do Ambiente e Execução

### Pré-requisitos
Certifique-se de possuir o Python 3.8+ instalado. Recomenda-se a instalação da distribuição com suporte a módulos extras do OpenCV (`opencv-contrib-python`), necessária para os módulos de reconhecimento facial avançados:

```bash
pip install opencv-contrib-python numpy matplotlib
```

*(Nota: caso seu ambiente possua apenas o `opencv-python` padrão, o software aciona automaticamente o classificador LBPH em NumPy, operando sem falhas).*

### Execução no VS Code ou Terminal

1. **Modo Interativo (Menu Completo):**
   ```bash
   python processamento_imagem_app.py
   ```
   Navegue pelas opções de 1 a 8 para selecionar qualquer operação, carregar imagens próprias ou interagir com o sistema.

2. **Modo Demonstração Automatizado (Apresentação Rápida):**
   ```bash
   python main.py --demo
   ```
   Gera dinamicamente uma base de dados sintética de sala de aula, executa todas as técnicas do enunciado em sequência e salva todas as imagens e relatórios de resultado na pasta `dados_teste/`.

---

## 3. Detalhamento do Desafio: Chamada Automatizada

O módulo prático implementa o fluxo de chamada inteligente inspirado no trabalho de **Mollosi et al. (2025)**:

1. **Cadastro dos Alunos:** Utiliza o arquivo `alunos_cadastrados.csv` mapeando matrícula, nome e imagem facial de referência.
2. **Treinamento do Classificador:** Extrai histogramas LBP de cada aluno cadastrado e treina o modelo preditivo.
3. **Captura e Detecção em Sala:**
   - **Modo Fotografia:** Processa uma imagem panorâmica da sala de aula, localiza as coordenadas faciais via Haar Cascade e classifica cada face.
   - **Modo Vídeo / Webcam:** Abre o feed ao vivo (`cv2.VideoCapture`), rastreia e identifica os alunos frame a frame (pressione a tecla `q` na janela de vídeo para encerrar e consolidar a ata).
4. **Exportação da Ata de Presença (CSV):**
   Gera automaticamente um relatório estruturado no arquivo `chamada_presenca.csv` com as colunas:
   - `Data`: Data da realização da chamada (YYYY-MM-DD).
   - `Horario`: Horário do registro de presença.
   - `ID_Matricula`: Código identificador único do aluno.
   - `Nome_Aluno`: Nome completo do estudante.
   - `Status_Presenca`: "Presente" ou "Ausente".
   - `Confianca`: Percentual estatístico de certeza do reconhecimento facial.

---

## 4. Referências Bibliográficas

- **Antonello, A.** – *Introdução à Visão Computacional com Python e OpenCV 3*. 2017. [Disponível online](https://www.google.com/search?q=https%3A%2F%2Fcv.antonello.com.br%2Fwp-content%2Fuploads%2F2017%2F02%2FLivro-Introdu%25C3%25A7%25C3%25A3o-a-Vis%25C3%25A3o-Computacional-com-Python-e-OpenCV-3.pdf).
- **Barelli, F.** – *Introdução à Visão Computacional: Uma abordagem prática com Python e OpenCV*. São Paulo: Casa do Código, 2018. ISBN: 9788594188588.
- **Mollosi, M. et al.** – *Sistema de Detecção e Contagem de Alunos em Sala de Aula utilizando Inteligência Artificial e Processamento de Vídeo*. Iguazu Science, v. 3, n. 7, p. 107-115, 2025. [Disponível online](https://www.google.com/search?q=https%3A%2F%2Fpublicacoes.uniguacu.com.br%2Findex.php%2Figuazu%2Farticle%2Fview%2F173%2F133).
- **Gonzalez, R. C.; Woods, R. E.** – *Processamento de Imagens Digitais*. São Paulo: Editora Blucher, 2000.
