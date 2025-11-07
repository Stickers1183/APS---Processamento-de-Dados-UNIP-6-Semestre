import numpy as np
import pandas as pd
try:
    import imageio.v2 as imageio
except ImportError:
    import imageio
import math
import cv2
import os
import warnings

# --- Filtros de Aviso ---
warnings.filterwarnings("ignore", message=".*hierarchy.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in true_divide.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in divide.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in double_scalars.*")

# --- Constantes Globais ---
ATTRIBUTE_INDEXES = [
    'palm_length', 'palm_width', 'hand_length', 'hand_width',
    'finger0_length', 'finger0_bot_width', 'finger0_mid_width', 'finger0_top_width',
    'finger1_length', 'finger1_bot_width', 'finger1_mid_width', 'finger1_top_width',
    'finger2_length', 'finger2_bot_width', 'finger2_mid_width', 'finger2_top_width',
    'finger3_length', 'finger3_bot_width', 'finger3_mid_width', 'finger3_top_width'
]
NUM_ATTRIBUTES = len(ATTRIBUTE_INDEXES)
CSV_COLUMN_NAMES = ATTRIBUTE_INDEXES + ['foto_id', 'person']

# --- Funções Geométricas e Auxiliares ---
def weigh(M, x1, x2, hist):
    x1 = max(0, x1)
    x2 = min(len(hist), x2)
    if x1 >= x2: return 0
    sum_hist = np.sum(hist[x1:x2]); return (1/M) * sum_hist if M > 0 else 0

def mean(x1, x2, hist):
    x1 = max(0, x1)
    x2 = min(len(hist), x2)
    if x1 >= x2: return 0
    a = sum(i * hist[i] for i in range(x1, x2)); sum_hist = np.sum(hist[x1:x2]); return a / sum_hist if sum_hist > 0 else 0

def variance(x1, x2, mean_val, hist):
    x1 = max(0, x1)
    x2 = min(len(hist), x2)
    if x1 >= x2: return 0
    if np.isnan(mean_val): return 0
    a = sum(math.pow(i - mean_val, 2) * hist[i] for i in range(x1, x2)); sum_hist = np.sum(hist[x1:x2]); return a / sum_hist if sum_hist > 0 else 0

def distance(p1, p2):
    if p1 is None or p2 is None: return 0
    try: return math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
    except (TypeError, IndexError): return 0

# --- Funções de Processamento de Imagem (Núcleo PIVC) ---
def grayTransform(img):
    if len(img.shape) == 2: return img.astype(float)
    if img.shape[2] == 4: img = img[:, :, :3]
    return np.dot(img.astype(float), [0.299, 0.587, 0.114])

def otsuThresholding(img):
    img_uint8 = img.astype(np.uint8)
    threshold_value, _ = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshold_value

def binaryTransform(img, thresholding):
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    _, binImg_cv = cv2.threshold(img_uint8, thresholding, 1, cv2.THRESH_BINARY)
    return binImg_cv.astype(float)

def selectBiggestObject(binImg):
    binary = (binImg * 255).astype(np.uint8)
    try: contours_data = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except: contours_data = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_data[0] if len(contours_data) == 2 else contours_data[1]
    if not contours: return (np.zeros_like(binary, dtype=float), None)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 100]
    if not valid_contours: return (np.zeros_like(binary, dtype=float), None)
    cnt = max(valid_contours, key=cv2.contourArea)
    mask = np.zeros_like(binary); cv2.drawContours(mask, [cnt], -1, (255), -1)
    return (mask / 255.0, cnt)

def findFingerPointsAndDefects(contour, filename_prefix=""):
    if contour is None or len(contour) < 5: return [], [], [], np.array([])
    try:
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        if hull_indices is None or not isinstance(hull_indices, np.ndarray) or hull_indices.ndim != 2 or hull_indices.shape[0] <= 3 or hull_indices.shape[1] != 1: return [], [], [], np.array([])
        hull_points = [tuple(contour[i][0]) for i in hull_indices.flatten()]
        defects_raw = cv2.convexityDefects(contour, hull_indices)
    except cv2.error as e: print(f" AvisoCV2(Hull/Defects):{e}", end=''); return [], [], [], np.array([])
    if defects_raw is None: print(f" Aviso:0 defeitos", end=''); return hull_points, [], [], np.array([])
    
    finger_valleys = []; filtered_defects_struct = []
    angle_threshold = 95; distance_threshold_raw = 4000
    
    for i in range(defects_raw.shape[0]):
        s_idx, e_idx, f_idx, d_raw = defects_raw[i, 0]
        if d_raw < distance_threshold_raw: continue
        if not (0 <= s_idx < len(contour) and 0 <= e_idx < len(contour) and 0 <= f_idx < len(contour)): continue
        start_pt = tuple(contour[s_idx][0]); end_pt = tuple(contour[e_idx][0]); far_pt = tuple(contour[f_idx][0])
        v1 = np.array(start_pt) - np.array(far_pt); v2 = np.array(end_pt) - np.array(far_pt)
        norm_v1 = np.linalg.norm(v1); norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0: angle = 180
        else: cosine_angle = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0); angle = math.degrees(math.acos(cosine_angle))
        if angle < angle_threshold: finger_valleys.append(far_pt); filtered_defects_struct.append(defects_raw[i])
    
    finger_tips = []
    min_dist_to_valley = 20
    for hp in hull_points:
        is_tip = all(distance(hp, fv) >= min_dist_to_valley for fv in finger_valleys)
        if is_tip: finger_tips.append(hp)
    
    num_filtered = len(finger_valleys)
    if num_filtered < 2 : print(f" Aviso:<2 vales({num_filtered})", end='')
    print(f" Info:{len(hull_points)} hull,{num_filtered} vales,{len(finger_tips)} pontas", end='')
    
    return hull_points, finger_valleys, finger_tips, np.array(filtered_defects_struct)

def cutPalm(mask_in, cnt, defects_struct, gray_img=None, output_dir=None, filename_prefix=""):
    mask = (mask_in * 255).astype(np.uint8)
    if defects_struct.shape[0] < 2: print(f" Aviso:Defeitos insuf({defects_struct.shape[0]}) p/ palma", end=''); return (mask_in, np.zeros_like(mask_in))
    points = []; f_idx_errors = 0
    for i in range(defects_struct.shape[0]):
        f_idx = defects_struct[i, 0, 2]
        if 0 <= f_idx < len(cnt): points.append(tuple(cnt[f_idx][0]))
        else: f_idx_errors += 1
    if f_idx_errors > 0: print(f" Aviso:Índices f_idx inválidos({f_idx_errors})", end='')
    if len(points) < 1 : print(f" Aviso:Pontos inválidos p/ minCirc", end=''); return (mask_in, np.zeros_like(mask_in))
    
    array = np.array(points, dtype=np.float32)
    try: (x, y), radius = cv2.minEnclosingCircle(array); center = (int(x), int(y)); radius = int(radius)
    except cv2.error as e: print(f" Aviso:Erro CV2(minCirc):{e}", end=''); return (mask_in, np.zeros_like(mask_in))
    
    cut_hand_mask_display = np.copy(mask); cv2.circle(cut_hand_mask_display, center, radius, (0), -1)
    palm_mask = np.zeros_like(mask); cv2.circle(palm_mask, center, radius, (255), -1)
    palm_mask = cv2.bitwise_and(palm_mask, mask)
    
    return (mask_in, palm_mask / 255.0)

def measure_fingers_new(mask_in, finger_tips, finger_valleys, filename_prefix=""):
    finger_measures = []; mask_h, mask_w = mask_in.shape[:2]
    
    # MUDANÇA: Verifica se há vales/pontas suficientes ANTES de tentar medi-los
    if len(finger_valleys) < 3 or len(finger_tips) < 4:
        print(f" Aviso:Pts insuf p/ medir 4 dedos(V:{len(finger_valleys)},P:{len(finger_tips)})", end='')
        # Em vez de retornar 0s, agora retorna None para indicar falha
        return None 
    
    finger_valleys.sort(key=lambda p: p[0]); finger_tips.sort(key=lambda p: p[0])
    possible_finger_valleys = sorted([v for v in finger_valleys], key=lambda p: p[1])
    if len(possible_finger_valleys) >= 5: possible_finger_valleys = possible_finger_valleys[:-2]
    elif len(possible_finger_valleys) == 4: possible_finger_valleys = possible_finger_valleys[:-1]
    possible_finger_valleys.sort(key=lambda p: p[0])
    
    if len(possible_finger_valleys) < 3:
         print(f" Aviso:<3 vales sup({len(possible_finger_valleys)}). Usando todos {len(finger_valleys)}.", end='')
         possible_finger_valleys = sorted(finger_valleys, key=lambda p: p[0])
    
    main_finger_tips = sorted(finger_tips, key=lambda p: p[0])
    if len(main_finger_tips) > 4: print(f" Info:Removendo ponta esq(polegar?)", end=''); main_finger_tips = main_finger_tips[1:5]
    elif len(main_finger_tips) < 4: print(f" Aviso:<4 pontas({len(main_finger_tips)})", end='')
    
    num_fingers_to_measure = min(len(main_finger_tips), len(possible_finger_valleys) + 1, 4)
    print(f" Info:Medindo {num_fingers_to_measure} dedos", end='')
    
    for i in range(num_fingers_to_measure):
        tip = main_finger_tips[i]
        if i == 0:
            if len(possible_finger_valleys) >= 1: valley1 = possible_finger_valleys[0]; valley2 = possible_finger_valleys[1] if len(possible_finger_valleys) > 1 else valley1
            else: valley1 = valley2 = (tip[0], mask_h-1)
        elif i < len(possible_finger_valleys): valley1 = possible_finger_valleys[i-1]; valley2 = possible_finger_valleys[i]
        else:
            if len(possible_finger_valleys) >= 2: valley1 = possible_finger_valleys[-2]; valley2 = possible_finger_valleys[-1]
            elif len(possible_finger_valleys) == 1: valley1 = valley2 = possible_finger_valleys[0]
            else: valley1 = valley2 = (tip[0], mask_h-1)
        
        base_point = (int((valley1[0] + valley2[0]) / 2), int((valley1[1] + valley2[1]) / 2))
        length = distance(tip, base_point)
        # NOTA: Esta é a versão do código que não rejeita dedos curtos, apenas os ignora.
        if length < 10: print(f" Aviso:Dedo{i} curto({length:.0f})", end=''); finger_measures.append((0, 0, 0, 0)); continue
        
        dx = tip[0] - base_point[0]; dy = tip[1] - base_point[1]; length_safe = length if length > 0 else 1.0
        
        def get_width_at_percent(percent):
            px = base_point[0] + percent * dx; py = base_point[1] + percent * dy
            perp_dx = -dy / length_safe; perp_dy = dx / length_safe; width_count = 0; max_search = int(mask_w / 3)
            cx_int, cy_int = int(px), int(py)
            if 0 <= cy_int < mask_h and 0 <= cx_int < mask_w and mask_in[cy_int, cx_int] > 0:
                 width_count = 1
                 for side in [-1, 1]:
                     for k in range(1, max_search):
                         x = int(px + side * k * perp_dx); y = int(py + side * k * perp_dy)
                         if 0 <= y < mask_h and 0 <= x < mask_w and mask_in[y, x] > 0: width_count += 1
                         else: break
            return width_count
        
        bot_width = get_width_at_percent(0.25); mid_width = get_width_at_percent(0.50); top_width = get_width_at_percent(0.75)
        finger_measures.append((int(length), bot_width, mid_width, top_width))
    
    while len(finger_measures) < 4: finger_measures.append((0, 0, 0, 0))
    return finger_measures[:4]

def objMeasure(obj_mask_in):
    obj_mask = (obj_mask_in * 255).astype(np.uint8)
    contours_data = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_data) == 2: contours = contours_data[0]
    else: contours = contours_data[1]
    if not contours: return (0, 0)
    contours = [c for c in contours if cv2.contourArea(c) > 50]
    if not contours: return (0, 0)
    maxC = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(maxC)
    return (height, width)

def normalization(values, hand_length):
    if hand_length == 0: return [0.0] * len(values)
    return [v / hand_length for v in values]

def generateAttributesList(palm_measure, hand_measure, fingers_measures):
    hand_length = hand_measure[0] if hand_measure[0] > 0 else 1.0
    values = [palm_measure[0], palm_measure[1], hand_measure[0], hand_measure[1]]
    for i in range(4): values.extend(fingers_measures[i])
    if len(values) != NUM_ATTRIBUTES:
         print(f" ERRO:TamListaAttrIncorreto({len(values)})", end='')
         values = (values + [0]*NUM_ATTRIBUTES)[:NUM_ATTRIBUTES]
    return normalization(values, hand_length)

# --- Função Chave: Processamento de Imagem ---
def processImage(filename, output_dir=None):
    """
    Função principal de PIVC. Recebe um caminho de imagem, processa
    e tenta extrair as características biométricas (medidas).
    Retorna (features, name, photo_id) se for legível.
    Retorna (None, None, None) se a imagem for ilegível.
    """
    base_filename = os.path.basename(filename); filename_prefix = os.path.splitext(base_filename)[0]
    
    try: parts = filename_prefix.split('_'); name = parts[0]; photo_id = int(parts[1])
    except: 
        print(f" Aviso:Nome/IDInválido '{base_filename}'", end=''); 
        name = "unknown"; photo_id = 0
    
    plot_subdir = None
    if output_dir: 
        plot_subdir = os.path.join(output_dir, filename_prefix); os.makedirs(plot_subdir, exist_ok=True)
    
    try: original_image = imageio.imread(filename)
    except Exception as e: print(f" ErroLeitura:{e}", end=''); return None, None, None
    
    gray = grayTransform(original_image)
    threshold = otsuThresholding(gray)
    if threshold <= 5 or threshold >= 250: print(f" Aviso:ThreshExtremo({threshold})", end=''); threshold = 127
    
    binImg = binaryTransform(gray, threshold)
    mask, contour = selectBiggestObject(binImg)
    
    if contour is None: print(f" Aviso:NoContour {base_filename}", end=''); return None, name, photo_id
    
    hull_points, finger_valleys, finger_tips, filtered_defects_struct = findFingerPointsAndDefects(contour, filename_prefix)
    
    # NOTA: O filtro de mão esquerda que estávamos depurando foi removido
    # no código que você enviou, então ele não está incluído aqui.
    
    _, palm_mask = cutPalm(mask, contour, filtered_defects_struct, gray, plot_subdir, filename_prefix)
    
    # MUDANÇA: 'measure_fingers_new' agora retorna None em caso de falha
    fingers_measures = measure_fingers_new(mask, finger_tips, finger_valleys, filename_prefix)
    if fingers_measures is None:
        # Esta é a falha por "dedos colados" ou "dedos dobrados"
        print(f" ERRO:Falha na medição dos dedos. Imagem ilegível.", end='')
        return None, name, photo_id

    palm_measure = objMeasure(palm_mask)
    hand_measure = objMeasure(mask)
    
    # --- MUDANÇA: FILTRO DE SANIDADE ADICIONADO ---
    if hand_measure[0] == 0 or hand_measure[1] == 0 or palm_measure[0] == 0:
         print(f" ERRO:Medidas inválidas (Mão:{hand_measure}, Palma:{palm_measure}). Imagem ilegível.", end='')
         return None, name, photo_id # Falha -> Ilegível

    if len(fingers_measures) != 4: print(f" ERRO:MedidasDedo!=4({len(fingers_measures)})", end=''); return None, name, photo_id
    
    numeric_values = generateAttributesList(palm_measure, hand_measure, fingers_measures)
    if len(numeric_values) != NUM_ATTRIBUTES: print(f" ERRO:Num Attrs({len(numeric_values)})", end=''); return None, name, photo_id

    # --- MUDANÇA: CHECAGEM FINAL ---
    if all(v == 0.0 for v in numeric_values):
        print(f" ERRO:Todas as features são zero. Imagem ilegível.", end='')
        return None, name, photo_id # Falha -> Ilegível
            
    return numeric_values, name, photo_id

# --- Funções de Interface para a Aplicação ---

def load_database(csv_path='measures.csv'):
    """
    Carrega o 'measures.csv' para um DataFrame do Pandas.
    """
    if not os.path.exists(csv_path):
        print(f"Erro: Arquivo de banco de dados não encontrado: {csv_path}")
        return pd.DataFrame()
    try:
        # Força o Pandas a ler a coluna 'person' como string (texto)
        data_df = pd.read_csv(csv_path, dtype={'person': str})
        
        data_df[ATTRIBUTE_INDEXES] = data_df[ATTRIBUTE_INDEXES].apply(pd.to_numeric, errors='coerce').fillna(0)
        return data_df
    except Exception as e:
        print(f"Erro ao carregar CSV: {e}")
        return pd.DataFrame()

def identify_user(image_path, database_df, threshold):
    """
    Lógica de Autenticação (1:N).
    Compara uma nova imagem com TODOS os registros do banco de dados.
    Retorna o 'person_id' (ex: "01") da melhor correspondência.
    """
    
    print(f"Processando imagem para autenticação: {image_path}")
    
    try:
        # Tenta processar a imagem
        new_measures, _, _ = processImage(image_path, output_dir=None) 
        
        if new_measures is None:
            # processImage falhou (mão esquerda, dedos colados, etc.)
            print("Não foi possível extrair características da imagem (ilegível).")
            return None, float('inf') # Retorna (None, inf) para sinalizar ILEGÍVEL
        
        new_measures_np = np.array(new_measures).astype(float)
        
        if np.any(np.isnan(new_measures_np)) or np.any(np.isinf(new_measures_np)):
             print("Características extraídas são inválidas (NaN/Inf).")
             return None, float('inf') # ILEGÍVEL
    except Exception as e:
        print(f"Erro ao processar imagem de autenticação: {e}")
        return None, float('inf') # ILEGÍVEL

    # MUDANÇA: Se o DB estiver vazio, a imagem é legível mas não há match.
    if database_df.empty:
        print("Banco de dados vazio, mas imagem é legível. Nenhum match.")
        return None, float('inf') - 1 # Retorna um valor diferente de 'inf'

    min_dist = float('inf')
    match_person = None
    
    # Compara com o banco de dados
    db_features = database_df[ATTRIBUTE_INDEXES].values
    db_persons = database_df['person'].values
    
    for i in range(len(db_features)):
        try:
            dist_test = np.linalg.norm(new_measures_np - db_features[i])
            
            if dist_test < min_dist:
                min_dist = dist_test
                match_person = db_persons[i]
        except Exception as e:
            print(f"Erro no cálculo da distância para o índice {i}: {e}")
            continue

    if min_dist < threshold:
        print(f"Match encontrado: {match_person} com distância {min_dist}")
        return match_person, min_dist
    else:
        print(f"Nenhum match encontrado. Distância mínima: {min_dist} (Threshold: {threshold})")
        return None, min_dist # Retorna (None, dist) para sinalizar NÃO ENCONTRADO
    