import os
import torch
import numpy as np
import cv2
from typing import Tuple, List
from pathlib import Path

# Imports específicos de tu framework (ajusta según tu estructura real)
from imashrimp_mmcv.mmcv.parallel import MMDataParallel
from imashrimp_mmcv.mmcv.runner import load_checkpoint
from imashrimp_ViTPose.mmpose.models import build_posenet
from imashrimp_ViTPose.mmpose.datasets.pipelines import Compose

# Optimizaciones opcionales
try:
    from mmcv.cnn import fuse_conv_bn
except ImportError:
    fuse_conv_bn = None


class PoseEstimationInferenceEngine:
    def __init__(self, checkpoint_path, general_cfg, device_id=0):
        """
        Inicializa el motor de inferencia. Carga modelo y pesos una sola vez.
        """
        self.device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        self.cfg = general_cfg

        # 1. Configuración del Modelo
        # Forzamos test_mode en la config del modelo si es necesario
        self.cfg.model.pretrained = None
        if self.cfg.data.test.get('test_mode', False) is False:
            self.cfg.data.test.test_mode = True

        print(f"[PoseEngine] Cargando modelo desde: {checkpoint_path}")

        # 2. Construir Modelo
        self.model = build_posenet(self.cfg.model)

        # 3. Cargar Pesos (Checkpoint)
        # map_location='cpu' es más seguro para evitar picos de VRAM al cargar, luego movemos.
        load_checkpoint(self.model, checkpoint_path, map_location='cpu')

        # 4. Optimizaciones Críticas de Velocidad
        # Fusión de Batch Normalization y Convolución (reduce operaciones)
        if self.cfg.get('fuse_conv_bn', False) and fuse_conv_bn:
            print("[PoseEngine] Fusionando Conv+BN para velocidad...")
            self.model = fuse_conv_bn(self.model)

        # Soporte FP16 (Half Precision) - Acelera dramáticamente en GPUs modernas (T4, V100, A100, RTX) FORZAR
        fp16_cfg = True#self.cfg.get('fp16', None)
        if fp16_cfg is not None:
            from imashrimp_mmcv.mmcv.runner import wrap_fp16_model
            wrap_fp16_model(self.model)
            self.is_fp16 = True
            print("[PoseEngine] Modo FP16 activado.")

        # 5. Mover a GPU y poner en modo evaluación
        self.model.to(self.device)
        self.model.eval()

        # 6. Construir Pipeline de Preprocesamiento (Sin DataLoader)
        # Extraemos el pipeline de test de la configuración
        # Esto evita crear el dataset object que es lento.
        test_pipeline = self.cfg.test_pipeline
        test_pipeline = test_pipeline[1:]
        self.transform_pipeline = Compose(test_pipeline)

        print("[PoseEngine] Motor listo para inferencia.")

        # Warmup (Opcional pero recomendado): Ejecutar una inferencia dummy para iniciar CUDA kernels
        self._warmup()

        self.bbox = [0, 0, 1920, 1080]
        self.center, self.scale = self._box_to_center_scale(self.bbox)
        self.joints_3d = np.zeros((37, 3), dtype=np.float32)
        self.joints_3d_visible = np.zeros((37, 3), dtype=np.float32)
        self.ann_info = self.cfg.data.test.get('data_cfg')
        if 'flip_pairs' not in self.ann_info:
            self.ann_info['flip_pairs'] = []

    def _warmup(self):
        """Ejecuta una pasada vacía para inicializar buffers de CUDA."""
        dummy_input = torch.zeros((1, 3, 256, 192)).to(self.device)  # Ajustar tamaño según config
        with torch.no_grad():
            try:
                # Intento genérico, depende de la estructura exacta del forward del modelo
                self.model.forward_dummy(dummy_input)
            except:
                pass

    @staticmethod
    def _box_to_center_scale(bbox, pixel_std=200):
        """Convierte [x, y, w, h] a centro y escala para mmpose."""
        x, y, w, h = bbox
        center = np.zeros(2, dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        # Mantener el aspect ratio de la red (normalmente definido en el config del modelo)
        # Si no, un cálculo estándar es:
        scale = np.array([w / pixel_std, h / pixel_std], dtype=np.float32)
        return center, scale

    @staticmethod
    def _read_line(path):
        with open(path, 'r') as f:
            try:
                linea = f.readline()
            except Exception as e:
                return ""
        return linea

    def _read_depth_file(self, path):
        try:
            d_path = self._read_line(path)
            if os.path.exists(d_path):
                return np.load(d_path)
            else:
                return np.load(path)
        except IOError as e:
            raise ValueError(f'Fail to read {path}: {e}')
            return None

    def _process_depth(self, dth_path):
        """Carga, normaliza y formatea el canal de profundidad."""
        try:
            # Carga rápida
            depth_image = self._read_depth_file(dth_path).astype(np.float32)

            # Normalización vectorizada (evita bucles lentos)
            mask = depth_image != 0
            if mask.any():
                min_val = np.min(depth_image[mask])
                max_val = np.max(depth_image)
                if max_val > min_val:
                    depth_image = (depth_image - min_val) / (max_val - min_val) * 255.0

            np.clip(depth_image, 0, None, out=depth_image)
            return np.expand_dims(depth_image, axis=2)  # (H, W, 1)
        except Exception as e:
            print(f"Error reading depth {dth_path}: {e}")
            # Retorna canal vacío negro en caso de fallo para no romper el pipeline
            return np.zeros((1080, 1920, 1), dtype=np.float32)  # Ajusta resolución si es fija

    def preprocess(self, img_rgb_numpy, dth_path):
        """
        1. Fusiona RGB + Depth.
        2. Ejecuta transformaciones de MMPose (CPU).
        3. Sube a GPU.

        Returns:
            input_data (tuple): (img_tensor, img_metas) listo para inferencia.
        """
        # 1. Preparar Depth
        depth_arr = self._process_depth(dth_path)

        # 2. Concatenar (H, W, 4)
        # Aseguramos float32 para la red
        input_data = np.concatenate((img_rgb_numpy.astype(np.float32), depth_arr), axis=2)

        # 3. Construir Dict para MMPose
        data = {
            'img': input_data,
            'img_shape': input_data.shape,
            'ori_shape': input_data.shape,
            'image_file': '',
            'depth_file': dth_path,
            'center': self.center,
            'scale': self.scale,
            'bbox': self.bbox,
            'rotation': 0,
            'dataset': 'camaron',
            'bbox_score': 1,
            'bbox_id': 0,
            'joints_3d': self.joints_3d,
            'joints_3d_visible': self.joints_3d_visible,
            'flip_pairs': self.ann_info['flip_pairs'],
            'ann_info': self.ann_info
        }

        # 4. Pipeline CPU (Resize, Normalize, ToTensor)
        # Esto convierte 'img' en un DC (DataContainer) con un Tensor dentro
        data = self.transform_pipeline(data)

        # 5. Extracción y Transferencia a GPU
        # data['img'] suele ser un DataContainer, accedemos a .data
        # Si ToTensor no envuelve en DataContainer, ajusta a data['img']
        tensor = data['img'].data if hasattr(data['img'], 'data') else data['img']

        # Añadir batch dim: (C, H, W) -> (1, C, H, W)
        tensor = tensor.unsqueeze(0)

        # Transferencia Asíncrona a GPU
        tensor = tensor.to(self.device, non_blocking=True)

        # Casting a FP16 si el modelo lo requiere
        if self.is_fp16:
            tensor = tensor.half()

        # Metadatos necesarios para el forward
        img_metas = [data['img_metas'].data] if hasattr(data['img_metas'], 'data') else [data['img_metas']]

        return tensor, img_metas

    def predict_async(self, input_data):
        """
        Ejecuta la inferencia en el stream actual.
        No bloquea la CPU esperando el resultado.

        Args:
            input_data: Tupla (tensor, metas) devuelta por preprocess.
        """
        img_tensor, img_metas = input_data

        # El contexto del stream (cuda.stream(s)) debe ser manejado fuera,
        # o asumimos el stream por defecto si no se especifica.
        # Aquí solo ejecutamos el forward.

        # MMPose forward_test suele ser: model(return_loss=False, ...)
        output = self.model(img=img_tensor, img_metas=img_metas, return_loss=False)

        return output

    def visualize_keypoints_and_bbox(self,
            img: np.ndarray,
            pov: int,
            ri: int,
            keypoints: np.ndarray,
            output_path: str,
            image_pth: str
    ) -> None:
        """
        Procesa una imagen para filtrar keypoints, calcular bounding box con padding
        y visualizar los resultados con etiquetas de clasificación.

        Args:
            img (np.ndarray): Imagen RGB de entrada (H, W, 3).
            pov (int): Punto de vista (0: Lateral, 1: Dorsal).
            ri (int): Integridad del Rostrum (0: Broken, 1: Good).
            keypoints (np.ndarray): Matriz de keypoints de forma (N, 3) -> [x, y, score].
            output_path (str): Ruta del directorio donde se guardará la imagen procesada.

        Raises:
            ValueError: Si los inputs de pov o ri no son válidos.
            IOError: Si hay problemas al guardar la imagen.
        """

        # --- 0. Validación y Preparación ---
        if img is None or img.size == 0:
            raise ValueError("La imagen proporcionada es inválida o está vacía.")

        # Copia de la imagen para no modificar la referencia original (buena práctica)
        vis_img = img.copy()
        h_img, w_img = vis_img.shape[:2]

        # Mapeo de textos para visualización
        pov_map = {0: "Lateral", 1: "Dorsal"}
        ri_map = {0: "Broken", 1: "Good"}

        if pov not in pov_map or ri not in ri_map:
            raise ValueError("Valores de 'pov' o 'ri' fuera de rango.")

        # --- 1. Lógica de Filtrado (Vectorizada) ---
        # Creamos una máscara booleana inicializada en True (todos válidos)
        num_kps = keypoints.shape[0]
        valid_mask = np.ones(num_kps, dtype=bool)

        # Definición de índices a excluir usando sets para búsqueda rápida O(1)
        indices_to_exclude = set()

        # Reglas de POV
        if pov == 0:  # Lateral: Ignorar 23-37
            indices_to_exclude.update(range(23, 38))
        elif pov == 1:  # Dorsal: Ignorar 9-22
            indices_to_exclude.update(range(9, 23))

        # Reglas de Rostrum
        if ri == 0:  # Broken: Ignorar índice 0 adicionalmente
            indices_to_exclude.add(0)

        # Aplicar exclusión a la máscara usando NumPy
        # Convertimos el set a lista para indexación de numpy
        if indices_to_exclude:
            exclude_arr = np.array(list(indices_to_exclude))
            # Aseguramos que los índices no excedan la cantidad de keypoints disponibles
            exclude_arr = exclude_arr[exclude_arr < num_kps]
            valid_mask[exclude_arr] = False

        # Filtramos los keypoints: Mantenemos la forma (M, 3)
        valid_kps = keypoints[valid_mask]

        # Si no quedan keypoints válidos, guardamos la imagen tal cual y salimos
        if valid_kps.shape[0] == 0:
            print("Advertencia: No hay keypoints válidos tras el filtrado.")
            self._save_image(vis_img, output_path, image_pth)
            return

        # --- 2. Cálculo del Bounding Box (BBox) ---
        # Usamos operaciones min/max de NumPy (mucho más rápido que bucles)
        # Coordenadas X e Y (asumiendo formato [x, y, score])
        x_coords = valid_kps[:, 0]
        y_coords = valid_kps[:, 1]

        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)

        padding = 20

        # Aplicamos padding y clipping (Clamp) para asegurar que esté dentro de la imagen
        # np.clip es altamente eficiente para esta operación
        x1 = int(np.clip(x_min - padding, 0, w_img))
        y1 = int(np.clip(y_min - padding, 0, h_img))
        x2 = int(np.clip(x_max + padding, 0, w_img))
        y2 = int(np.clip(y_max + padding, 0, h_img))

        # --- 3. Visualización (Drawing) ---

        # A. Dibujar Keypoints
        # Iterar sobre los puntos filtrados.
        # Nota: cv2.circle no acepta arrays vectorizados directamente para múltiples centros,
        # pero el overhead aquí es mínimo ya que M (puntos válidos) es pequeño.
        for kp in valid_kps:
            x, y = int(kp[0]), int(kp[1])
            # Dibujamos círculo verde (BGR: 0, 255, 0)
            cv2.circle(vis_img, (x, y), 4, (0, 255, 0), -1)

        # B. Dibujar Bounding Box
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Rojo, grosor 2

        # C. Etiqueta de Texto con Fondo
        label_text = f"{pov_map[pov]} - {ri_map[ri]}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text_color = (255, 255, 255)  # Blanco
        bg_color = (0, 0, 255)  # Rojo (coincide con el bbox)

        # Calcular tamaño del texto para el fondo
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

        # Coordenadas del fondo del texto (encima del bbox)
        # Si el bbox está muy arriba, dibujamos el texto por dentro para que no se corte
        text_x = x1
        text_y = y1 - 5 if y1 - 5 > text_h + 5 else y1 + text_h + 10

        cv2.rectangle(vis_img,
                      (text_x, text_y - text_h - 5),
                      (text_x + text_w, text_y + baseline - 5),
                      bg_color,
                      -1)

        cv2.putText(vis_img, label_text, (text_x, text_y - 5), font, font_scale, text_color, font_thickness)

        # --- 4. Salida ---
        self._save_image(vis_img, output_path, image_pth)

    def _save_image(self, img: np.ndarray, output_dir: str, image_pth: int) -> None:
        """Helper privado para manejar la escritura de archivos y directorios."""
        try:
            # Crear directorio si no existe (equivalente a mkdir -p)
            save_dir = Path(output_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Generar nombre de archivo único o descriptivo
            filename = f"{image_pth}"
            full_path = save_dir / filename

            success = cv2.imwrite(str(full_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if not success:
                raise IOError(f"OpenCV falló al escribir la imagen en {full_path}")

            print(f"Proceso completado. Imagen guardada en: {full_path}")

        except Exception as e:
            print(f"Error crítico al guardar la imagen: {e}")