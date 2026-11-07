import os
import sys
import csv
import collections
from datetime import datetime

import cv2
import numpy as np
import matplotlib
# Configuração segura de backend para ambientes gráficos ou headless
try:
    if os.environ.get('DISPLAY') is None and sys.platform != 'win32' and sys.platform != 'darwin':
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    pass

# MÓDULO 1: OPERAÇÕES DE ENTRADA/SAÍDA E VISUALIZAÇÃO
class ImageIOManager:
    """Gerencia leitura, duplicação, exibição simultânea e gravação de imagens."""

    @staticmethod
    def load_image(filepath: str) -> np.ndarray:
        """Abre e carrega uma imagem a partir de um arquivo em disco."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
        img = cv2.imread(filepath)
        if img is None:
            raise ValueError(f"Não foi possível decodificar a imagem: {filepath}")
        return img

    @staticmethod
    def duplicate_image(img: np.ndarray) -> np.ndarray:
        """Cria uma cópia profunda (independente em memória) da imagem."""
        return img.copy()

    @staticmethod
    def save_image(img: np.ndarray, filepath: str) -> bool:
        """Salva a imagem processada em arquivo."""
        folder = os.path.dirname(os.path.abspath(filepath))
        if folder:
            os.makedirs(folder, exist_ok=True)
        return cv2.imwrite(filepath, img)

    @staticmethod
    def show_side_by_side(original: np.ndarray, processed: np.ndarray, 
                          title_orig: str = "Imagem Original", 
                          title_proc: str = "Imagem Processada",
                          save_preview_path: str = "preview_comparacao.png") -> None:
        """
        Exibe simultaneamente a imagem original e a imagem processada lado a lado.
        Suporta tanto escala de cinza quanto imagem colorida (BGR).
        Salva também um arquivo de preview caso o ambiente não possua display gráfico.
        """
        if len(original.shape) == 3:
            orig_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            cmap_orig = None
        else:
            orig_rgb = original
            cmap_orig = 'gray'

        if len(processed.shape) == 3:
            proc_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            cmap_proc = None
        else:
            proc_rgb = processed
            cmap_proc = 'gray'

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        axes[0].imshow(orig_rgb, cmap=cmap_orig)
        axes[0].set_title(title_orig, fontsize=12, fontweight='bold', color='#1a365d')
        axes[0].axis('off')

        axes[1].imshow(proc_rgb, cmap=cmap_proc)
        axes[1].set_title(title_proc, fontsize=12, fontweight='bold', color='#2b6cb0')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(save_preview_path, dpi=150, bbox_inches='tight')
        
        try:
            if matplotlib.get_backend().lower() != 'agg':
                plt.show(block=False)
                plt.pause(0.5)
        except Exception:
            pass
        plt.close(fig)
        print(f"-> Visualização comparativa salva em: {save_preview_path}")

# MÓDULO 2: OPERAÇÕES BÁSICAS EM IMAGENS
class BasicOperations:
    """Implementa transformações pontuais, geométricas e conversões de espaço de cor."""

    @staticmethod
    def invert_colors(img: np.ndarray) -> np.ndarray:
        """Inversão de cores (Negativo da imagem)."""
        return cv2.bitwise_not(img)

    @staticmethod
    def adjust_brightness_contrast(img: np.ndarray, alpha: float = 1.2, beta: int = 30) -> np.ndarray:
        """
        Ajuste linear de contraste (alpha > 0) e brilho (beta: [-100, 100]).
        Fórmula: g(x) = alpha * f(x) + beta com saturação [0, 255].
        """
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    @staticmethod
    def add_images(img1: np.ndarray, img2: np.ndarray, weight1: float = 0.5, weight2: float = 0.5) -> np.ndarray:
        """
        Adição e fusão ponderada de duas imagens.
        Redimensiona automaticamente img2 para casar com as dimensões de img1.
        """
        h, w = img1.shape[:2]
        img2_resized = cv2.resize(img2, (w, h))
        if len(img1.shape) == 3 and len(img2_resized.shape) == 2:
            img2_resized = cv2.cvtColor(img2_resized, cv2.COLOR_GRAY2BGR)
        elif len(img1.shape) == 2 and len(img2_resized.shape) == 3:
            img2_resized = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
        return cv2.addWeighted(img1, weight1, img2_resized, weight2, 0.0)

    @staticmethod
    def subtract_images(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Subtração com saturação: cv2.subtract(img1, img2)."""
        h, w = img1.shape[:2]
        img2_resized = cv2.resize(img2, (w, h))
        if len(img1.shape) != len(img2_resized.shape):
            if len(img1.shape) == 3:
                img2_resized = cv2.cvtColor(img2_resized, cv2.COLOR_GRAY2BGR)
            else:
                img2_resized = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)
        return cv2.subtract(img1, img2_resized)

    @staticmethod
    def to_grayscale(img: np.ndarray) -> np.ndarray:
        """Conversão para níveis de cinza (Y = 0.299 R + 0.587 G + 0.114 B)."""
        if len(img.shape) == 2:
            return img.copy()
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def convert_bgr_hsv(img: np.ndarray, to_hsv: bool = True) -> np.ndarray:
        """Conversão entre modelos de cor BGR e HSV."""
        if to_hsv:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        else:
            return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    @staticmethod
    def resize_image(img: np.ndarray, width: int = None, height: int = None, scale: float = None) -> np.ndarray:
        """Redimensionamento por proporção (scale) ou dimensões fixas (largura x altura)."""
        h, w = img.shape[:2]
        if scale is not None:
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        elif width is not None and height is not None:
            new_w, new_h = width, height
        elif width is not None:
            ratio = width / float(w)
            new_w, new_h = width, max(1, int(h * ratio))
        elif height is not None:
            ratio = height / float(h)
            new_w, new_h = max(1, int(w * ratio)), height
        else:
            return img.copy()
        
        interp = cv2.INTER_AREA if (new_w < w or new_h < h) else cv2.INTER_CUBIC
        return cv2.resize(img, (new_w, new_h), interpolation=interp)

    @staticmethod
    def crop_roi(img: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
        """Recorte de uma Região de Interesse (Region of Interest - ROI)."""
        h_img, w_img = img.shape[:2]
        x1 = max(0, min(x, w_img - 1))
        y1 = max(0, min(y, h_img - 1))
        x2 = max(x1 + 1, min(x + width, w_img))
        y2 = max(y1 + 1, min(y + height, h_img))
        return img[y1:y2, x1:x2].copy()

# MÓDULO 3: FILTRAGEM ESPACIAL E REALCE
class FilteringAndEnhancement:
    """Implementa filtros passa-baixas (suavização) e passa-altas (bordas)."""

    @staticmethod
    def mean_filter(img: np.ndarray, ksize: int = 5) -> np.ndarray:
        """Filtro da média (Box Filter). Suavização linear uniforme."""
        ksize = max(1, ksize if ksize % 2 == 1 else ksize + 1)
        return cv2.blur(img, (ksize, ksize))

    @staticmethod
    def median_filter(img: np.ndarray, ksize: int = 5) -> np.ndarray:
        """Filtro da mediana. Não-linear, altamente eficaz contra ruído sal e pimenta."""
        ksize = max(1, ksize if ksize % 2 == 1 else ksize + 1)
        return cv2.medianBlur(img, ksize)

    @staticmethod
    def gaussian_filter(img: np.ndarray, ksize: int = 5, sigma: float = 1.0) -> np.ndarray:
        """Filtro Gaussiano. Suavização ponderada por distribuição normal bidimensional."""
        ksize = max(1, ksize if ksize % 2 == 1 else ksize + 1)
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)

    @staticmethod
    def sobel_edges(img: np.ndarray, ksize: int = 3) -> np.ndarray:
        """
        Detecção de bordas pelo operador Sobel.
        Calcula gradientes Gx e Gy e retorna a magnitude combinada.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        ksize = max(1, ksize if ksize % 2 == 1 else ksize + 1)
        
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        
        magnitude = cv2.magnitude(grad_x, grad_y)
        return np.clip(magnitude, 0, 255).astype(np.uint8)

    @staticmethod
    def canny_edges(img: np.ndarray, threshold1: float = 100, threshold2: float = 200) -> np.ndarray:
        """
        Algoritmo ótimo de detecção de bordas de Canny.
        Inclui suavização gaussiana, cálculo de gradiente, supressão de não-máximos e histerese.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        return cv2.Canny(gray, threshold1, threshold2)

# MÓDULO 4: HISTOGRAMA, INTENSIDADE E QUANTIZAÇÃO
class HistogramAndIntensity:
    """Operações estatísticas no domínio espacial, canais de cor e quantização."""

    @staticmethod
    def plot_histogram(img: np.ndarray, save_path: str = "histograma.png") -> None:
        """Cria e salva a curva de distribuição do histograma (cinza ou BGR)."""
        fig = plt.figure(figsize=(8, 4))
        if len(img.shape) == 2:
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            plt.plot(hist, color='#2d3748', lw=1.8, label='Intensidade (Cinza)')
            plt.fill_between(range(256), hist.ravel(), color='#a0aec0', alpha=0.3)
        else:
            colors = ('b', 'g', 'r')
            labels = ('Canal Azul (B)', 'Canal Verde (G)', 'Canal Vermelho (R)')
            for i, col in enumerate(colors):
                hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                plt.plot(hist, color=col, lw=1.5, label=labels[i])
        
        plt.title("Histograma de Frequência de Intensidades", fontsize=12, fontweight='bold')
        plt.xlabel("Nível de Intensidade (0 - 255)")
        plt.ylabel("Contagem de Pixels")
        plt.xlim([0, 256])
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"-> Histograma salvo em: {save_path}")

    @staticmethod
    def equalize_histogram(img: np.ndarray) -> np.ndarray:
        """
        Equalização de histograma.
        Se colorida, equaliza o canal de Luminância/Valor (V) no modelo HSV
        para preservar a fidelidade de matiz e saturação sem distorcer cores.
        """
        if len(img.shape) == 2:
            return cv2.equalizeHist(img)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def split_channels(img: np.ndarray) -> tuple:
        """Separação (splitting) nos canais fundamentais B, G e R."""
        if len(img.shape) == 2:
            return img, img, img
        return cv2.split(img)

    @staticmethod
    def binarize(img: np.ndarray, method: str = 'otsu', fixed_thresh: int = 127) -> np.ndarray:
        """
        Binarização segundo limiar:
        - 'fixed'     : Limiar global fixo (fixed_thresh)
        - 'adaptive'  : Limiar local adaptativo (Gaussian C)
        - 'otsu'      : Algoritmo de Otsu (minimiza variância intra-classe)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        
        if method.lower() == 'fixed':
            _, binary = cv2.threshold(gray, fixed_thresh, 255, cv2.THRESH_BINARY)
        elif method.lower() == 'adaptive':
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
        elif method.lower() == 'otsu':
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            raise ValueError(f"Método de binarização desconhecido: {method}")
        return binary

    @staticmethod
    def quantize_grayscale(img: np.ndarray, levels: int = 4) -> np.ndarray:
        """Redução controlada dos níveis de cinza (quantização uniforme)."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        levels = max(2, min(levels, 256))
        step = 256.0 / levels
        quantized = (np.floor(gray / step) * step + (step / 2.0)).clip(0, 255)
        return quantized.astype(np.uint8)

    @staticmethod
    def quantize_colors_kmeans(img: np.ndarray, k: int = 8) -> np.ndarray:
        """Quantização de cores utilizando agrupamento K-Means (OpenCV)."""
        if len(img.shape) == 2:
            return HistogramAndIntensity.quantize_grayscale(img, levels=k)
        
        data = img.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()].reshape(img.shape)
        return quantized

# MÓDULO 5: SEGMENTAÇÃO E MORFOLOGIA MATEMÁTICA
class SegmentationAndMorphology:
    """Algoritmo de Crescimento de Regiões e Operações Morfológicas Estruturais."""

    @staticmethod
    def region_growing(img: np.ndarray, seed_point: tuple = None, threshold: int = 18) -> np.ndarray:
        """
        Segmentação por Crescimento de Regiões (Region Growing).
        Implementado com fila FIFO (conectividade 8) conforme Gonzalez & Woods.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        h, w = gray.shape

        if seed_point is None:
            seed_point = (w // 2, h // 2)

        xs, ys = seed_point
        xs = max(0, min(w - 1, xs))
        ys = max(0, min(h - 1, ys))

        segmented = np.zeros((h, w), dtype=np.uint8)
        visited = np.zeros((h, w), dtype=bool)

        queue = collections.deque([(xs, ys)])
        visited[ys, xs] = True
        segmented[ys, xs] = 255

        region_sum = float(gray[ys, xs])
        region_count = 1

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]

        while queue:
            cx, cy = queue.popleft()
            current_mean = region_sum / region_count

            for dx, dy in neighbors:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                    visited[ny, nx] = True
                    val = float(gray[ny, nx])
                    if abs(val - current_mean) <= threshold:
                        segmented[ny, nx] = 255
                        region_sum += val
                        region_count += 1
                        queue.append((nx, ny))

        return segmented

    @staticmethod
    def morphology_op(img: np.ndarray, operation: str = 'opening',
                      ksize: int = 5, shape: str = 'rect', iterations: int = 1) -> np.ndarray:
        """
        Operações morfológicas matemáticas:
        - 'erosion'   : Erosão (contração de regiões claras)
        - 'dilation'  : Dilatação (expansão de regiões claras)
        - 'opening'   : Abertura (erosão seguida de dilatação, remove ruídos finos)
        - 'closing'   : Fechamento (dilatação seguida de erosão, preenche buracos)
        """
        shape_dict = {
            'rect': cv2.MORPH_RECT,
            'ellipse': cv2.MORPH_ELLIPSE,
            'cross': cv2.MORPH_CROSS
        }
        elem_shape = shape_dict.get(shape.lower(), cv2.MORPH_RECT)
        kernel = cv2.getStructuringElement(elem_shape, (ksize, ksize))

        op_lower = operation.lower()
        if op_lower == 'erosion':
            return cv2.erode(img, kernel, iterations=iterations)
        elif op_lower == 'dilation':
            return cv2.dilate(img, kernel, iterations=iterations)
        elif op_lower == 'opening':
            return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif op_lower == 'closing':
            return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        else:
            raise ValueError(f"Operação morfológica inválida: {operation}")

# MÓDULO 6: RECONHECIMENTO FACIAL (ANTONELLO / HAAR + LBPH)
class LBPHFallbackClassifier:
    """
    Extrator e classificador LBPH em Python/NumPy puro.
    Garante compatibilidade total caso o ambiente execute opencv-python básico
    sem o pacote opencv-contrib-python.
    """
    def __init__(self, grid_x=8, grid_y=8):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.faces_data = [] # Lista de tuplas: (vetor_caracteristicas, id_aluno)

    def extract_features(self, gray_face: np.ndarray) -> np.ndarray:
        if len(gray_face.shape) == 3:
            gray_face = cv2.cvtColor(gray_face, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray_face, (128, 128))
        h, w = resized.shape
        lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
        center = resized[1:-1, 1:-1]
        
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
                   (1, 1), (1, 0), (1, -1), (0, -1)]
        for i, (dy, dx) in enumerate(offsets):
            neighbor = resized[1+dy:h-1+dy, 1+dx:w-1+dx]
            lbp += ((neighbor >= center) << i).astype(np.uint8)

        lh, lw = lbp.shape
        ch, cw = lh // self.grid_y, lw // self.grid_x
        hists = []
        for gy in range(self.grid_y):
            for gx in range(self.grid_x):
                cell = lbp[gy*ch:(gy+1)*ch, gx*cw:(gx+1)*cw]
                hist, _ = np.histogram(cell.ravel(), bins=64, range=(0, 256))
                hist = hist.astype(np.float32)
                hist /= (np.linalg.norm(hist) + 1e-6)
                hists.append(hist)
        return np.concatenate(hists)

    def train(self, face_list: list, labels: list) -> None:
        self.faces_data = []
        for face, lbl in zip(face_list, labels):
            feat = self.extract_features(face)
            self.faces_data.append((feat, lbl))

    def predict(self, face: np.ndarray) -> tuple:
        if not self.faces_data:
            return -1, 999.0
        query_feat = self.extract_features(face)
        best_dist = float('inf')
        best_lbl = -1
        for train_feat, lbl in self.faces_data:
            eps = 1e-10
            chi2 = 0.5 * np.sum(((query_feat - train_feat) ** 2) / (query_feat + train_feat + eps))
            if chi2 < best_dist:
                best_dist = chi2
                best_lbl = lbl
        norm_dist = float(best_dist * 10.0)
        return best_lbl, norm_dist

class FaceRecognitionSystem:
    """
    Sistema de detecção e reconhecimento facial baseado na metodologia de
    Antonello (2017) utilizando Haar Cascade e classificador LBPH.
    """
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.student_db = {} # ID (int) -> Nome (str)
        self.trained = False

        if hasattr(cv2, 'face') and hasattr(cv2.face, 'LBPHFaceRecognizer_create'):
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.use_native_lbph = True
        else:
            self.recognizer = LBPHFallbackClassifier()
            self.use_native_lbph = False

    def detect_faces(self, img: np.ndarray) -> list:
        """
        Detecta coordenadas (x, y, w, h) de faces com busca adaptativa
        para garantir sensibilidade tanto em fotos reais quanto sintetizadas.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
        )
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=2, minSize=(25, 25)
            )
        return list(faces)

    def train_students(self, faces_list: list, ids_list: list, names_dict: dict) -> None:
        """Treina o modelo com a base de dados de faces dos alunos."""
        if not faces_list or not ids_list:
            raise ValueError("Lista de faces ou de IDs vazia para treinamento.")
        
        self.student_db = names_dict
        gray_faces = []
        for face in faces_list:
            if len(face.shape) == 3:
                gray_faces.append(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY))
            else:
                gray_faces.append(face)

        if self.use_native_lbph:
            self.recognizer.train(gray_faces, np.array(ids_list, dtype=np.int32))
        else:
            self.recognizer.train(gray_faces, ids_list)

        self.trained = True

    def recognize_face(self, face_crop: np.ndarray) -> tuple:
        """Retorna (id_aluno, nome_aluno, confianca_percentual)."""
        if not self.trained:
            raise RuntimeError("O modelo de reconhecimento precisa ser treinado antes da predição.")

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if len(face_crop.shape) == 3 else face_crop
        label_id, distance = self.recognizer.predict(gray)

        confidence_pct = max(0.0, min(100.0, 100.0 - (distance - 30.0)))
        name = self.student_db.get(label_id, "Desconhecido")
        
        if distance > 85.0:
            name = "Desconhecido"

        return label_id, name, confidence_pct

# MÓDULO 7: DESAFIO ADICIONAL - CHAMADA AUTOMATIZADA EM SALA DE AULA
class AutomatedAttendanceSystem:
    """
    Implementação prática do desafio adicional:
    Módulo de chamada automatizada capaz de detectar alunos em sala de aula,
    reconhecê-los e emitir ata de presença em formato CSV (Mollosi et al., 2025).
    """
    def __init__(self, face_system: FaceRecognitionSystem):
        self.face_system = face_system

    def process_classroom_image(self, classroom_img: np.ndarray, 
                                output_annotated_path: str = "chamada_detectada.png",
                                csv_report_path: str = "chamada_presenca.csv") -> dict:
        """
        Processa a imagem da sala de aula, anota as caixas delimitadoras e emite o relatório CSV.
        """
        annotated = classroom_img.copy()
        faces = self.face_system.detect_faces(classroom_img)
        present_students = {}

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        for (x, y, w, h) in faces:
            face_roi = classroom_img[y:y+h, x:x+w]
            if face_roi.size == 0:
                continue

            try:
                sid, name, conf = self.face_system.recognize_face(face_roi)
            except Exception:
                sid, name, conf = -1, "Desconhecido", 0.0

            if name != "Desconhecido":
                present_students[sid] = {
                    'name': name,
                    'confidence': conf,
                    'time': time_str
                }
                color = (0, 200, 0) # Verde para reconhecido
                label_text = f"{name} ({conf:.1f}%)"
            else:
                color = (0, 0, 255) # Vermelho para desconhecido
                label_text = "Desconhecido"

            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(annotated, (x, y - 24), (x + w, y), color, -1)
            cv2.putText(annotated, label_text, (x + 5, y - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imwrite(output_annotated_path, annotated)

        all_registered = self.face_system.student_db
        with open(csv_report_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Data", "Horario", "ID_Matricula", "Nome_Aluno", "Status_Presenca", "Confianca"])
            for sid, sname in all_registered.items():
                if sid in present_students:
                    status = "Presente"
                    conf_str = f"{present_students[sid]['confidence']:.1f}%"
                else:
                    status = "Ausente"
                    conf_str = "0.0%"
                writer.writerow([date_str, time_str, sid, sname, status, conf_str])

        summary = {
            'total_registered': len(all_registered),
            'present_count': len(present_students),
            'absent_count': len(all_registered) - len(present_students),
            'detected_faces': len(faces),
            'present_students': present_students,
            'annotated_image': output_annotated_path,
            'csv_report': csv_report_path
        }
        return summary

    def process_live_video(self, video_source=0, csv_report_path: str = "chamada_presenca_video.csv"):
        """
        Processa feed de vídeo ou webcam em tempo real para chamada dinâmica.
        Encerra ao pressionar 'q'.
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"[!] Não foi possível abrir o dispositivo de vídeo: {video_source}")
            return None

        print("[*] Iniciando captura de vídeo. Pressione 'q' na janela para encerrar.")
        present_students = {}
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                faces = self.face_system.detect_faces(frame)
                for (x, y, w, h) in faces:
                    roi = frame[y:y+h, x:x+w]
                    if roi.size == 0:
                        continue
                    sid, name, conf = self.face_system.recognize_face(roi)
                    if name != "Desconhecido":
                        present_students[sid] = {
                            'name': name,
                            'confidence': conf,
                            'time': datetime.now().strftime("%H:%M:%S")
                        }
                        color = (0, 255, 0)
                        lbl = f"{name} ({conf:.0f}%)"
                    else:
                        color = (0, 0, 255)
                        lbl = "Desconhecido"
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, lbl, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                cv2.imshow("Chamada Automatizada - Video Stream", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"[!] Exibição interativa não suportada no terminal atual: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()

        with open(csv_report_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Data", "Horario", "ID_Matricula", "Nome_Aluno", "Status_Presenca", "Confianca"])
            for sid, sname in self.face_system.student_db.items():
                if sid in present_students:
                    writer.writerow([date_str, present_students[sid]['time'], sid, sname, "Presente", f"{present_students[sid]['confidence']:.1f}%"])
                else:
                    writer.writerow([date_str, "-", sid, sname, "Ausente", "0.0%"])

        print(f"[✓] Chamada por vídeo concluída. Ata salva em: {csv_report_path}")

# UTILITÁRIOS E GERADOR DE DADOS SINTÉTICOS PARA DEMONSTRAÇÃO
def generate_synthetic_classroom_and_students():
    """
    Gera dinamicamente imagens de teste realistas com faces estilizadas
    para permitir execução e testes imediatos mesmo sem webcam ou arquivos prévios.
    """
    os.makedirs("dados_teste/alunos", exist_ok=True)
    students = {
        1: "Ana Silva",
        2: "Carlos Eduardo",
        3: "Beatriz Santos",
        4: "Lucas Mendes"
    }

    trained_faces = []
    trained_ids = []

    for sid, name in students.items():
        face_img = np.full((120, 120, 3), 220, dtype=np.uint8)
        skin_tone = (180 + sid*8, 200 + sid*5, 230)
        cv2.ellipse(face_img, (60, 65), (42, 50), 0, 0, 360, skin_tone, -1)
        
        # Sobrancelhas e olhos com alto contraste (reconhecível por Haar)
        cv2.line(face_img, (38, 48), (56, 48), (30, 20, 10), 2)
        cv2.line(face_img, (64, 48), (82, 48), (30, 20, 10), 2)
        cv2.circle(face_img, (47, 56), 6, (20, 20, 20), -1)
        cv2.circle(face_img, (73, 56), 6, (20, 20, 20), -1)
        # Sombra do nariz e boca
        cv2.line(face_img, (60, 58), (60, 74), (120, 130, 150), 2)
        cv2.rectangle(face_img, (48, 86), (72, 92), (50, 50, 160), -1)

        # Cabelo e acessórios distintos
        if sid == 1: # Ana: Cabelo castanho comprido
            cv2.ellipse(face_img, (60, 32), (46, 20), 0, 0, 360, (20, 40, 90), -1)
            cv2.rectangle(face_img, (15, 32), (32, 100), (20, 40, 90), -1)
            cv2.rectangle(face_img, (88, 32), (105, 100), (20, 40, 90), -1)
        elif sid == 2: # Carlos: Cabelo curto escuro
            cv2.rectangle(face_img, (18, 15), (102, 35), (25, 25, 25), -1)
        elif sid == 3: # Beatriz: Cabelo claro volumoso
            cv2.circle(face_img, (60, 30), 40, (30, 80, 140), -1)
        else: # Lucas: Cabelo escuro e óculos
            cv2.ellipse(face_img, (60, 25), (42, 16), 0, 0, 360, (10, 10, 10), -1)
            cv2.rectangle(face_img, (38, 49), (56, 63), (60, 60, 60), 2)
            cv2.rectangle(face_img, (64, 49), (82, 63), (60, 60, 60), 2)
            cv2.line(face_img, (56, 56), (64, 56), (60, 60, 60), 2)

        path = f"dados_teste/alunos/aluno_{sid}.png"
        cv2.imwrite(path, face_img)
        trained_faces.append(face_img)
        trained_ids.append(sid)

    classroom = np.full((360, 600, 3), 240, dtype=np.uint8)
    cv2.rectangle(classroom, (50, 20), (550, 120), (50, 100, 50), -1)
    cv2.putText(classroom, "Visao Computacional - PDI", (140, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Posiciona 3 alunos nas carteiras (Ana=1, Carlos=2, Lucas=4; Beatriz=3 ausente)
    positions = [(80, 180, 1), (240, 180, 2), (400, 180, 4)]
    for x, y, sid in positions:
        face = trained_faces[sid - 1]
        classroom[y:y+120, x:x+120] = face

    classroom_path = "dados_teste/sala_de_aula.png"
    cv2.imwrite(classroom_path, classroom)
    return students, trained_faces, trained_ids, classroom_path

# INTERFACE DO MENU MODULAR PRINCIPAL
class PDIFinalProjectApp:
    """Controlador central da aplicação e menu interativo."""

    def __init__(self):
        self.io = ImageIOManager()
        self.basic = BasicOperations()
        self.filter = FilteringAndEnhancement()
        self.intensity = HistogramAndIntensity()
        self.segment = SegmentationAndMorphology()
        self.face_sys = FaceRecognitionSystem()
        self.attendance = AutomatedAttendanceSystem(self.face_sys)

        self.current_img = None
        self.original_img = None
        self.current_path = ""

    def ensure_image_loaded(self) -> bool:
        """Verifica se há uma imagem carregada para processamento."""
        if self.current_img is None:
            print("\n[!] Nenhuma imagem carregada na memória.")
            print("Carregue uma imagem no Menu 1 ou gere a imagem de demonstração.")
            return False
        return True

    def run_automated_demo(self):
        """Executa um pipeline de demonstração cobrindo 100% dos requisitos."""
        print("\n" + "="*65)
        print("  EXECUTANDO DEMONSTRAÇÃO COMPLETA DE TODAS AS TÉCNICAS")
        print("="*65)
        
        students, faces, ids, class_path = generate_synthetic_classroom_and_students()
        img = self.io.load_image(class_path)
        self.original_img = self.io.duplicate_image(img)
        self.current_img = self.io.duplicate_image(img)
        print("[+] 1. Entrada/Saída: Imagem de teste carregada e duplicada com sucesso.")

        inv = self.basic.invert_colors(img)
        bright = self.basic.adjust_brightness_contrast(img, alpha=1.3, beta=40)
        gray = self.basic.to_grayscale(img)
        hsv = self.basic.convert_bgr_hsv(img, to_hsv=True)
        resized = self.basic.resize_image(img, scale=0.5)
        roi = self.basic.crop_roi(img, 80, 180, 120, 120)
        self.io.save_image(inv, "dados_teste/demo_invertida.png")
        self.io.save_image(bright, "dados_teste/demo_brilho.png")
        self.io.save_image(gray, "dados_teste/demo_cinza.png")
        print("[+] 2. Operações Básicas: Inversão, Brilho, Cinza, HSV, Redimensionamento e ROI executados.")

        mean_blur = self.filter.mean_filter(img, 7)
        median_blur = self.filter.median_filter(img, 5)
        gauss_blur = self.filter.gaussian_filter(img, 7, sigma=1.5)
        sobel = self.filter.sobel_edges(gray, 3)
        canny = self.filter.canny_edges(gray, 50, 150)
        self.io.save_image(sobel, "dados_teste/demo_sobel.png")
        self.io.save_image(canny, "dados_teste/demo_canny.png")
        print("[+] 3. Filtragem: Média, Mediana, Gaussiano, Sobel e Canny processados.")

        self.intensity.plot_histogram(img, "dados_teste/demo_histograma.png")
        equalized = self.intensity.equalize_histogram(img)
        b, g, r = self.intensity.split_channels(img)
        bin_otsu = self.intensity.binarize(gray, method='otsu')
        bin_adapt = self.intensity.binarize(gray, method='adaptive')
        quant_gray = self.intensity.quantize_grayscale(gray, levels=4)
        quant_col = self.intensity.quantize_colors_kmeans(img, k=6)
        self.io.save_image(equalized, "dados_teste/demo_equalizada.png")
        self.io.save_image(bin_otsu, "dados_teste/demo_binaria_otsu.png")
        self.io.save_image(quant_col, "dados_teste/demo_quantizada.png")
        print("[+] 4. Histograma: Equalização, Splitting BGR, Binarização Otsu/Adaptativa e Quantização gerados.")

        reg_grow = self.segment.region_growing(gray, seed_point=(140, 240), threshold=15)
        erosion = self.segment.morphology_op(bin_otsu, 'erosion', ksize=5)
        dilation = self.segment.morphology_op(bin_otsu, 'dilation', ksize=5)
        opening = self.segment.morphology_op(bin_otsu, 'opening', ksize=5)
        closing = self.segment.morphology_op(bin_otsu, 'closing', ksize=5)
        self.io.save_image(reg_grow, "dados_teste/demo_crescimento_regioes.png")
        self.io.save_image(opening, "dados_teste/demo_abertura.png")
        print("[+] 5. Segmentação e Morfologia: Crescimento de Regiões, Erosão, Dilatação, Abertura e Fechamento concluídos.")

        print("[+] 6 & 7. Treinando faces dos alunos e executando Chamada Automatizada...")
        self.face_sys.train_students(faces, ids, students)
        report = self.attendance.process_classroom_image(
            img, 
            output_annotated_path="dados_teste/chamada_resultado.png",
            csv_report_path="dados_teste/chamada_presenca.csv"
        )

        print("\n--- RELATÓRIO DA CHAMADA AUTOMATIZADA ---")
        print(f"Total de Alunos Matriculados: {report['total_registered']}")
        print(f"Alunos Presentes Detectados  : {report['present_count']}")
        print(f"Alunos Ausentes              : {report['absent_count']}")
        print(f"Imagem anotada salva em      : {report['annotated_image']}")
        print(f"Arquivo CSV de presença      : {report['csv_report']}")
        print("\nLista de Presença:")
        for sid, pdata in report['present_students'].items():
            print(f"  [✓] ID {sid}: {pdata['name']} (Confiança: {pdata['confidence']:.1f}%) às {pdata['time']}")
        print("="*65)
        print("DEMONSTRAÇÃO CONCLUÍDA COM SUCESSO! Todos os arquivos foram gerados em 'dados_teste/'.\n")

    def menu_loop(self):
        """Loop principal do menu interativo no terminal."""
        while True:
            status = "Carregada (" + str(self.current_img.shape) + ")" if self.current_img is not None else "Nenhuma"
            print("\n" + "="*60)
            print("  SISTEMA MODULAR DE PROCESSAMENTO DE IMAGENS E VÍDEO")
            print(f"  Imagem Atual: {status}")
            print("="*60)
            print("1. Operações de Entrada/Saída e Visualização")
            print("2. Operações Básicas em Imagens")
            print("3. Filtragem Espacial e Realce")
            print("4. Histogramas, Binarização e Quantização")
            print("5. Segmentação (Crescimento de Regiões) e Morfologia")
            print("6. Reconhecimento Facial (Antonello / LBPH)")
            print("7. Desafio Adicional: Chamada Automatizada (CSV / Vídeo)")
            print("8. Executar Demonstração Automatizada Completa")
            print("0. Sair")
            print("="*60)
            
            choice = input("Escolha uma opção (0-8): ").strip()
            if choice == '0':
                print("Encerrando o programa. Até logo!")
                break
            elif choice == '1':
                self._submenu_io()
            elif choice == '2':
                self._submenu_basic()
            elif choice == '3':
                self._submenu_filters()
            elif choice == '4':
                self._submenu_histogram()
            elif choice == '5':
                self._submenu_segmentation()
            elif choice == '6':
                self._submenu_face()
            elif choice == '7':
                self._submenu_attendance()
            elif choice == '8':
                self.run_automated_demo()
            else:
                print("[!] Opção inválida. Tente novamente.")

    def _submenu_io(self):
        print("\n--- [1] Operações de E/S e Visualização ---")
        print("1. Abrir/Ler imagem de arquivo")
        print("2. Duplicar imagem original (Resetar edições)")
        print("3. Mostrar imagem original e processada lado a lado")
        print("4. Gravar/Salvar imagem processada em arquivo")
        print("5. Carregar imagem sintética de teste (Sala de Aula)")
        print("0. Voltar")
        op = input("Opção: ").strip()
        if op == '1':
            path = input("Caminho do arquivo de imagem: ").strip()
            try:
                self.current_img = self.io.load_image(path)
                self.original_img = self.io.duplicate_image(self.current_img)
                self.current_path = path
                print(f"[✓] Imagem carregada com sucesso: {self.current_img.shape}")
            except Exception as e:
                print(f"[!] Erro ao carregar imagem: {e}")
        elif op == '2':
            if self.ensure_image_loaded():
                self.current_img = self.io.duplicate_image(self.original_img)
                print("[✓] Imagem resetada para a cópia original.")
        elif op == '3':
            if self.ensure_image_loaded():
                out_path = input("Caminho para salvar o preview [preview_comparacao.png]: ").strip() or "preview_comparacao.png"
                self.io.show_side_by_side(self.original_img, self.current_img, save_preview_path=out_path)
        elif op == '4':
            if self.ensure_image_loaded():
                out = input("Nome do arquivo de saída (ex: resultado.png): ").strip() or "resultado.png"
                self.io.save_image(self.current_img, out)
                print(f"[✓] Imagem salva em: {out}")
        elif op == '5':
            _, _, _, path = generate_synthetic_classroom_and_students()
            self.current_img = self.io.load_image(path)
            self.original_img = self.io.duplicate_image(self.current_img)
            self.current_path = path
            print(f"[✓] Imagem sintética de sala de aula carregada: {path}")

    def _submenu_basic(self):
        if not self.ensure_image_loaded():
            return
        print("\n--- [2] Operações Básicas ---")
        print("1. Inversão de cores (Negativo)")
        print("2. Ajuste de Brilho e Contraste")
        print("3. Adição/Fusão com outra imagem")
        print("4. Subtração de imagens")
        print("5. Conversão para níveis de cinza")
        print("6. Conversão RGB <-> HSV")
        print("7. Redimensionamento da imagem")
        print("8. Recorte de Região de Interesse (ROI)")
        print("0. Voltar")
        op = input("Opção: ").strip()
        if op == '1':
            self.current_img = self.basic.invert_colors(self.current_img)
            print("[✓] Cores invertidas com sucesso.")
        elif op == '2':
            alpha = float(input("Fator de contraste alpha (ex: 1.3): ") or 1.2)
            beta = int(input("Ajuste de brilho beta [-100 a 100] (ex: 30): ") or 30)
            self.current_img = self.basic.adjust_brightness_contrast(self.current_img, alpha, beta)
            print("[✓] Brilho e contraste ajustados.")
        elif op == '3':
            path2 = input("Caminho da segunda imagem: ").strip()
            try:
                img2 = self.io.load_image(path2)
                w1 = float(input("Peso da imagem 1 [0.0 a 1.0, padrão 0.5]: ") or 0.5)
                self.current_img = self.basic.add_images(self.current_img, img2, w1, 1.0 - w1)
                print("[✓] Imagens fundidas com sucesso.")
            except Exception as e:
                print(f"[!] Erro na adição: {e}")
        elif op == '4':
            path2 = input("Caminho da segunda imagem para subtrair: ").strip()
            try:
                img2 = self.io.load_image(path2)
                self.current_img = self.basic.subtract_images(self.current_img, img2)
                print("[✓] Subtração realizada.")
            except Exception as e:
                print(f"[!] Erro na subtração: {e}")
        elif op == '5':
            self.current_img = self.basic.to_grayscale(self.current_img)
            print("[✓] Convertida para níveis de cinza.")
        elif op == '6':
            if len(self.current_img.shape) == 3:
                hsv = self.basic.convert_bgr_hsv(self.current_img, to_hsv=True)
                print("[✓] Convertida para espaço de cor HSV.")
                self.current_img = hsv
            else:
                print("[!] A imagem precisa ser colorida para conversão HSV.")
        elif op == '7':
            sc = float(input("Fator de escala (ex: 0.5 para 50%, 2.0 para 200%): ") or 0.5)
            self.current_img = self.basic.resize_image(self.current_img, scale=sc)
            print(f"[✓] Redimensionada para: {self.current_img.shape}")
        elif op == '8':
            x = int(input("Coordenada X inicial: ") or 50)
            y = int(input("Coordenada Y inicial: ") or 50)
            w = int(input("Largura do recorte: ") or 150)
            h = int(input("Altura do recorte: ") or 150)
            self.current_img = self.basic.crop_roi(self.current_img, x, y, w, h)
            print(f"[✓] Região de interesse recortada: {self.current_img.shape}")

    def _submenu_filters(self):
        if not self.ensure_image_loaded():
            return
        print("\n--- [3] Filtragem Espacial e Realce ---")
        print("1. Filtro da Média (Box Blur)")
        print("2. Filtro da Mediana (Redução de ruído)")
        print("3. Filtro Gaussiano")
        print("4. Detector de Bordas Sobel")
        print("5. Detector de Bordas Canny")
        print("0. Voltar")
        op = input("Opção: ").strip()
        if op == '1':
            k = int(input("Tamanho do kernel ímpar (ex: 3, 5, 7): ") or 5)
            self.current_img = self.filter.mean_filter(self.current_img, k)
            print("[✓] Filtro da média aplicado.")
        elif op == '2':
            k = int(input("Tamanho do kernel ímpar (ex: 3, 5): ") or 5)
            self.current_img = self.filter.median_filter(self.current_img, k)
            print("[✓] Filtro da mediana aplicado.")
        elif op == '3':
            k = int(input("Tamanho do kernel (ex: 5): ") or 5)
            s = float(input("Desvio padrão sigma (ex: 1.5): ") or 1.5)
            self.current_img = self.filter.gaussian_filter(self.current_img, k, s)
            print("[✓] Filtro gaussiano aplicado.")
        elif op == '4':
            k = int(input("Tamanho do kernel Sobel (1, 3, 5): ") or 3)
            self.current_img = self.filter.sobel_edges(self.current_img, k)
            print("[✓] Bordas calculadas com operador de Sobel.")
        elif op == '5':
            t1 = float(input("Limiar inferior (ex: 100): ") or 100)
            t2 = float(input("Limiar superior (ex: 200): ") or 200)
            self.current_img = self.filter.canny_edges(self.current_img, t1, t2)
            print("[✓] Bordas extraídas com Canny.")

    def _submenu_histogram(self):
        if not self.ensure_image_loaded():
            return
        print("\n--- [4] Histogramas, Intensidade e Quantização ---")
        print("1. Gerar e exibir gráfico de histograma")
        print("2. Equalização de histograma")
        print("3. Separar (split) canais B, G, R")
        print("4. Binarização por limiar fixo")
        print("5. Binarização adaptativa")
        print("6. Binarização pelo método de Otsu")
        print("7. Quantização de níveis de cinza")
        print("8. Quantização de cores com K-Means")
        print("0. Voltar")
        op = input("Opção: ").strip()
        if op == '1':
            out = input("Arquivo para salvar gráfico [histograma.png]: ").strip() or "histograma.png"
            self.intensity.plot_histogram(self.current_img, out)
        elif op == '2':
            self.current_img = self.intensity.equalize_histogram(self.current_img)
            print("[✓] Histograma equalizado.")
        elif op == '3':
            b, g, r = self.intensity.split_channels(self.current_img)
            self.io.save_image(b, "canal_azul.png")
            self.io.save_image(g, "canal_verde.png")
            self.io.save_image(r, "canal_vermelho.png")
            print("[✓] Canais exportados individualmente: canal_azul.png, canal_verde.png, canal_vermelho.png.")
        elif op == '4':
            t = int(input("Limiar fixo [0-255] (ex: 127): ") or 127)
            self.current_img = self.intensity.binarize(self.current_img, method='fixed', fixed_thresh=t)
            print("[✓] Binarização com limiar fixo concluída.")
        elif op == '5':
            self.current_img = self.intensity.binarize(self.current_img, method='adaptive')
            print("[✓] Binarização adaptativa gaussiana aplicada.")
        elif op == '6':
            self.current_img = self.intensity.binarize(self.current_img, method='otsu')
            print("[✓] Binarização pelo método de Otsu concluída.")
        elif op == '7':
            n = int(input("Número de níveis de cinza (ex: 4, 8, 16): ") or 4)
            self.current_img = self.intensity.quantize_grayscale(self.current_img, levels=n)
            print(f"[✓] Quantização em {n} níveis aplicada.")
        elif op == '8':
            k = int(input("Número de cores K (ex: 4, 8, 16): ") or 8)
            self.current_img = self.intensity.quantize_colors_kmeans(self.current_img, k=k)
            print(f"[✓] Quantização K-Means ({k} cores) aplicada.")

    def _submenu_segmentation(self):
        if not self.ensure_image_loaded():
            return
        print("\n--- [5] Segmentação e Morfologia Matemática ---")
        print("1. Crescimento de Regiões (Region Growing)")
        print("2. Erosão Morfológica")
        print("3. Dilatação Morfológica")
        print("4. Abertura Morfológica")
        print("5. Fechamento Morfológico")
        print("0. Voltar")
        op = input("Opção: ").strip()
        if op == '1':
            h, w = self.current_img.shape[:2]
            xs = int(input(f"Coordenada X da semente [0 a {w-1}, padrão {w//2}]: ") or w//2)
            ys = int(input(f"Coordenada Y da semente [0 a {h-1}, padrão {h//2}]: ") or h//2)
            th = int(input("Limiar de tolerância de intensidade (ex: 15): ") or 15)
            self.current_img = self.segment.region_growing(self.current_img, (xs, ys), th)
            print("[✓] Segmentação por crescimento de regiões concluída.")
        elif op in ('2', '3', '4', '5'):
            op_map = {'2': 'erosion', '3': 'dilation', '4': 'opening', '5': 'closing'}
            k = int(input("Tamanho do kernel morfológico (ex: 3, 5): ") or 5)
            shape = input("Formato do elemento [rect, ellipse, cross]: ").strip() or "rect"
            it = int(input("Número de iterações: ") or 1)
            self.current_img = self.segment.morphology_op(self.current_img, op_map[op], k, shape, it)
            print(f"[✓] Operação morfológica '{op_map[op]}' aplicada com sucesso.")

    def _submenu_face(self):
        print("\n--- [6] Reconhecimento Facial (Antonello / LBPH) ---")
        print("1. Detectar faces na imagem atual (Haar Cascade)")
        print("2. Treinar base de alunos sintética/exemplo")
        print("3. Reconhecer faces na imagem atual")
        print("0. Voltar")
        op = input("Opção: ").strip()
        if op == '1':
            if not self.ensure_image_loaded():
                return
            faces = self.face_sys.detect_faces(self.current_img)
            annotated = self.current_img.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
            self.io.save_image(annotated, "faces_detectadas.png")
            print(f"[✓] {len(faces)} face(s) detectada(s). Resultado salvo em: faces_detectadas.png")
        elif op == '2':
            students, faces, ids, _ = generate_synthetic_classroom_and_students()
            self.face_sys.train_students(faces, ids, students)
            print(f"[✓] Treinamento concluído com sucesso para {len(students)} alunos:")
            for sid, sname in students.items():
                print(f"    - ID {sid}: {sname}")
        elif op == '3':
            if not self.ensure_image_loaded():
                return
            if not self.face_sys.trained:
                print("[!] O modelo precisa ser treinado primeiro. Use a opção 2.")
                return
            faces = self.face_sys.detect_faces(self.current_img)
            annotated = self.current_img.copy()
            for (x, y, w, h) in faces:
                face_crop = self.current_img[y:y+h, x:x+w]
                sid, name, conf = self.face_sys.recognize_face(face_crop)
                cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(annotated, f"{name} ({conf:.0f}%)", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            self.io.save_image(annotated, "reconhecimento_resultado.png")
            print(f"[✓] Processamento concluído. Salvo em: reconhecimento_resultado.png")

    def _submenu_attendance(self):
        print("\n--- [7] Desafio Adicional: Chamada Automatizada em Sala de Aula ---")
        print("1. Processar foto de sala de aula e gerar Ata de Presença (CSV)")
        print("2. Iniciar Chamada por Vídeo / Webcam em Tempo Real")
        print("3. Visualizar última ata de presença gerada")
        print("0. Voltar")
        op = input("Opção: ").strip()
        if op == '1':
            if not self.face_sys.trained:
                print("[*] Base não treinada. Treinando base de alunos automaticamente...")
                students, faces, ids, _ = generate_synthetic_classroom_and_students()
                self.face_sys.train_students(faces, ids, students)

            path = input("Caminho da foto da sala [Enter para usar foto sintética]: ").strip()
            if not path or not os.path.exists(path):
                _, _, _, path = generate_synthetic_classroom_and_students()
                print(f"[*] Utilizando imagem sintética de sala: {path}")

            classroom_img = self.io.load_image(path)
            report = self.attendance.process_classroom_image(
                classroom_img,
                output_annotated_path="chamada_sala_anotada.png",
                csv_report_path="chamada_presenca.csv"
            )
            print("\n" + "="*50)
            print("  ATA DE CHAMADA GERADA COM SUCESSO")
            print("="*50)
            print(f"Total de Alunos Matriculados: {report['total_registered']}")
            print(f"Alunos Presentes            : {report['present_count']}")
            print(f"Alunos Ausentes             : {report['absent_count']}")
            print(f"Imagem com marcações salvas : {report['annotated_image']}")
            print(f"Planilha de Presença (CSV)  : {report['csv_report']}")
            print("="*50)
        elif op == '2':
            if not self.face_sys.trained:
                print("[*] Treinando base de alunos para reconhecimento no vídeo...")
                students, faces, ids, _ = generate_synthetic_classroom_and_students()
                self.face_sys.train_students(faces, ids, students)
            source = input("Origem do vídeo [0 para Webcam ou caminho do arquivo .mp4]: ").strip()
            v_src = int(source) if source.isdigit() else (source if source else 0)
            self.attendance.process_live_video(video_source=v_src)
        elif op == '3':
            csv_path = "chamada_presenca.csv"
            if not os.path.exists(csv_path):
                print(f"[!] Arquivo '{csv_path}' ainda não foi gerado.")
                return
            print(f"\nConteúdo de {csv_path}:")
            print("-" * 65)
            with open(csv_path, mode='r', encoding='utf-8') as f:
                for line in f:
                    print(line.strip())
            print("-" * 65)

# PONTO DE ENTRADA DO SCRIPT
if __name__ == "__main__":
    app = PDIFinalProjectApp()
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        app.run_automated_demo()
    else:
        app.menu_loop()