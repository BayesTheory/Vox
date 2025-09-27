# UA-DETRAC -> YOLO (1 classe: car) para PROCESSED/UA_DETRAC_CAR
# - Foco total na classe "car" (ID original informável via CAR_ID_OVERRIDE)
# - Mantém apenas "car" e remapeia para classe 0 (zero-based, contínua)
# - Verifica/normaliza xywh para [0,1] e descarta boxes degenerados
# - Copia imagens por hardlink (quando possível) + fallback copy2
# - Aceita múltiplas extensões (.jpg, .jpeg, .png, .bmp)
# - Gera detrac_car.yaml no padrão Ultralytics e manifest JSON
# - Barras de progresso com tqdm

from pathlib import Path
from collections import Counter
import json
import shutil
import os
import cv2
from tqdm import tqdm

# CONFIG: ajuste as pastas de origem/destino
SRC_ROOT = Path(r"C:\Users\riana\OneDrive\Desktop\Vox MVP\data\UA-DETRAC\DETRAC_Upload")
DST_ROOT = Path(r"C:\Users\riana\OneDrive\Desktop\Vox MVP\data\PROCESSED\UA_DETRAC_CAR")

SRC_IM = {"train": SRC_ROOT/"images"/"train", "val": SRC_ROOT/"images"/"val"}
SRC_LB = {"train": SRC_ROOT/"labels"/"train", "val": SRC_ROOT/"labels"/"val"}

DST_IM = {"train": DST_ROOT/"images"/"train", "val": DST_ROOT/"images"/"val"}
DST_LB = {"train": DST_ROOT/"labels"/"train", "val": DST_ROOT/"labels"/"val"}

# Se souber o ID de 'car' nos rótulos originais, defina aqui (ex.: 3);
# caso contrário, deixe None para detecção automática pela classe mais frequente.
CAR_ID_OVERRIDE = 3  # defina 3 se sua taxonomia original for aquela lista com "car"=3

EXTS = (".jpg", ".jpeg", ".png", ".bmp")  # suportadas pelo loader do YOLO

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_label_file(p: Path):
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            try:
                c = int(parts[0]); x,y,w,h = map(float, parts[1:])
                rows.append([c,x,y,w,h])
            except Exception:
                pass
    return rows

def is_normalized_xywh(rows):
    for _, x, y, w, h in rows:
        if max(x, y, w, h) > 1.5:
            return False
    return True

def clamp01(v): 
    return max(0.0, min(1.0, v))

def xyxy_to_xywh_norm(x1, y1, x2, y2, W, H):
    xc = ((x1 + x2) / 2.0) / W
    yc = ((y1 + y2) / 2.0) / H
    ww = (abs(x2 - x1)) / W
    hh = (abs(y2 - y1)) / H
    return clamp01(xc), clamp01(yc), clamp01(ww), clamp01(hh)

# Normalização com W,H já conhecidos
def normalize_rows(rows, W, H):
    if not rows:
        return []
    if is_normalized_xywh(rows):
        return [[c, clamp01(x), clamp01(y), clamp01(w), clamp01(h)] for c,x,y,w,h in rows]
    nr = []
    for c, a, b, c3, d in rows:
        if c3 > 1.5 and d > 1.5:  # assume xywh em pixels
            xc = a / W; yc = b / H; ww = c3 / W; hh = d / H
        else:  # assume xyxy em pixels
            x1, y1, x2, y2 = a, b, c3, d
            xc, yc, ww, hh = xyxy_to_xywh_norm(x1, y1, x2, y2, W, H)
        nr.append([c, clamp01(xc), clamp01(yc), clamp01(ww), clamp01(hh)])
    return nr

def discover_car_id():
    cls_freq = Counter()
    all_labels = list(SRC_LB["train"].glob("*.txt")) + list(SRC_LB["val"].glob("*.txt"))
    for txt in tqdm(all_labels, desc="Descobrindo ID de 'car'"):
        for r in read_label_file(txt):
            cls_freq[r[0]] += 1
    if not cls_freq:
        return 0, {}
    car_id = cls_freq.most_common(1)[0][0]  # 'car' tende a ser o mais frequente
    return car_id, dict(cls_freq)

def copy_image(src_img: Path, dst_img: Path):
    ensure_dir(dst_img.parent)
    if dst_img.exists():
        return
    try:
        os.link(src_img, dst_img)  # hardlink (mesmo volume)
    except Exception:
        shutil.copy2(src_img, dst_img)  # fallback

def write_yaml(dst_root: Path):
    yaml = (
        f"path: {dst_root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\nnc: 1\n"
        f"names: [car]\n"
    )
    (dst_root/"detrac_car.yaml").write_text(yaml, encoding="utf-8")

def collect_images(images_dir: Path):
    paths = []
    for ext in EXTS:
        paths.extend(images_dir.rglob(f"*{ext}"))
    return sorted(paths)

def process_split(split, car_id: int, summary: dict):
    ensure_dir(DST_IM[split]); ensure_dir(DST_LB[split])
    stats = {"images": 0, "labels_in": 0, "boxes_total": 0, "boxes_kept": 0, "empties": 0}
    img_paths = collect_images(SRC_IM[split])

    # Cache de dimensões por pasta absoluta para evitar colisões
    dim_cache = {}

    for src_img in tqdm(img_paths, desc=f"Processando split '{split}'"):
        stem = src_img.stem
        src_lbl = SRC_LB[split] / f"{stem}.txt"
        
        rows = read_label_file(src_lbl)
        if rows:
            stats["labels_in"] += 1
        stats["boxes_total"] += len(rows)

        # Copia imagem (YOLO ignora imagens sem .txt)
        dst_img = DST_IM[split] / src_img.name
        copy_image(src_img, dst_img)
        stats["images"] += 1

        # Manter apenas car_id
        car_rows = [r for r in rows if int(r[0]) == car_id]
        if not car_rows:
            # garante que não haja .txt antigo no destino
            dst_lbl = DST_LB[split] / f"{stem}.txt"
            if dst_lbl.exists():
                dst_lbl.unlink()
            stats["empties"] += 1
            continue

        # Dimensões por cache
        seq_key = src_img.parent.as_posix()
        if seq_key not in dim_cache:
            img = cv2.imread(str(src_img))
            H, W = img.shape[:2]
            dim_cache[seq_key] = (W, H)
        else:
            W, H = dim_cache[seq_key]

        # Normalizar
        normalized = normalize_rows(car_rows, W, H)

        # Descartar boxes degenerados (área muito pequena / fora de [0,1] antes do clamp final)
        filtered = []
        for _, x, y, w, h in normalized:
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0 and (w * h) >= 1e-6:
                filtered.append((x, y, w, h))

        stats["boxes_kept"] += len(filtered)

        dst_lbl = DST_LB[split] / f"{stem}.txt"
        if filtered:
            with dst_lbl.open("w", encoding="utf-8") as f:
                for x, y, w, h in filtered:
                    f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        else:
            if dst_lbl.exists():
                dst_lbl.unlink()
            stats["empties"] += 1

    summary["splits"][split] = stats

def main():
    # Limpar destino para exportação consistente
    if DST_ROOT.exists():
        print(f"Pasta de destino {DST_ROOT} já existe. Limpando...")
        shutil.rmtree(DST_ROOT)
    ensure_dir(DST_ROOT)

    # Definir ou descobrir CAR_ID
    if CAR_ID_OVERRIDE is not None:
        car_id, freq = CAR_ID_OVERRIDE, {}
        print(f"Usando ID de 'car' pré-definido: {car_id}")
    else:
        print("Buscando ID da classe mais comum ('car')...")
        car_id, freq = discover_car_id()

    summary = {"car_id_used": car_id, "source_class_frequencies": freq, "splits": {}}

    # Processar splits
    for split in ("train", "val"):
        process_split(split, car_id, summary)

    # YAML e manifestos
    write_yaml(DST_ROOT)
    manifest_path = DST_ROOT / "ua_detrac_preprocess_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "="*50)
    print("Resumo do Processamento:")
    print(json.dumps(summary, indent=2))
    print(f"\n✅ Dataset pronto em: {DST_ROOT.resolve()}")
    print("="*50)

if __name__ == "__main__":
    main()
