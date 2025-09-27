# Pré-processamento com verificação e retomada (10 cores VeRi, sem limite)
# - Corrige XML gb2312 (gb18030/gbk -> UTF-8)
# - Filtra somente "carros" (VeRi) e somente as 10 cores do VeRi
# - Verifica pastas existentes, evita duplicatas e resume sem sobrescrever
# - Gera manifest/resumo por split/classe
# Saída: COLOR_FINAL_YOLO no formato Ultralytics (pastas por classe + data.yaml)

import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import shutil, json, re

# ---------- Constantes ----------
VERI_COLOR_ID = {1:"yellow",2:"orange",3:"green",4:"gray",5:"red",6:"blue",7:"white",8:"golden",9:"brown",10:"black"}  # cores VeRi [1]
CAR_TYPES = {1,2,4,5,9}  # sedan, suv, hatchback, mpv, estate [1]
COLOR_MAP = {
    "yellow":"amarelo","orange":"laranja","green":"verde","gray":"cinza_prata","red":"vermelho",
    "blue":"azul","white":"branco","golden":"dourado","brown":"marrom","black":"preto",
    "grey":"cinza_prata","silver":"cinza_prata","beige":"bege","gold":"dourado","purple":"roxo","pink":"rosa","tan":"bege"
}  # normalização p/ pastas por classe Ultralytics [1]
ALLOWED_10 = {"amarelo","laranja","verde","cinza_prata","vermelho","azul","branco","dourado","marrom","preto"}  # paleta VeRi [1]

# ---------- Utilidades de arquivo ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)  # cria diretório de forma idempotente [2]

def list_files(dirp: Path):
    return {f.name for f in dirp.glob("*.*")} if dirp.exists() else set()  # inventário rápido por pasta [2]

def safe_copy(src: Path, dst_dir: Path, prefix: str, allow_collision_suffix=True):
    """Copia evitando sobrescrever; se existir o mesmo nome, adiciona sufixo incremental."""
    if not src.exists():
        return None
    ensure_dir(dst_dir)
    base = f"{prefix}_{src.name}"
    dst = dst_dir / base
    if not dst.exists():
        shutil.copy2(src, dst)
        return dst
    if not allow_collision_suffix:
        return None
    # cria sufixos _1, _2, ...
    stem, suf = dst.stem, dst.suffix
    i = 1
    while True:
        cand = dst_dir / f"{stem}_{i}{suf}"
        if not cand.exists():
            shutil.copy2(src, cand)
            return cand
        i += 1  # caminho existe; tenta próximo [2]

def load_xml_root_utf8(xml_path: Path):
    """Lê XML VeRi declarados como gb2312/GBK/GB18030 e normaliza para UTF-8 (correção ElementTree)."""
    data = xml_path.read_bytes()
    text = None
    for enc in ("gb18030","gbk","utf-8"):
        try:
            text = data.decode(enc); break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = data.decode("latin-1", errors="ignore")
    if text.lstrip().startswith("<?xml"):
        text = re.sub(r'encoding=[\'\"].*?[\'\"]', 'encoding="utf-8"', text, count=1)
    else:
        text = '<?xml version="1.0" encoding="utf-8"?>\n' + text
    return ET.fromstring(text.encode("utf-8"))  # evita erro do Expat com multibyte [2]

def summarize_tree(root_dir: Path):
    """Resumo por split/classe: contagem de arquivos, útil para auditoria e retomada."""
    summary = defaultdict(lambda: defaultdict(int))
    if not root_dir.exists():
        return summary
    for split in root_dir.iterdir():
        if not split.is_dir():
            continue
        for cls in split.iterdir():
            if cls.is_dir():
                summary[split.name][cls.name] = sum(1 for _ in cls.glob("*.*"))
    return summary  # ajuda a validar formato Ultralytics (pastas por classe) [1]

def save_json(obj, path: Path):
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")  # persistência idempotente [2]

# ---------- Processamentos ----------
def process_veri(veri_root: Path, out_root: Path) -> dict:
    """Extrai APENAS carros do VeRi e organiza por cor normalizada nas 10 cores."""
    stats = {"total":0,"cars":0,"colors":defaultdict(int)}
    veri_out = out_root / "veri_processed"
    cfgs = [("train","train_label.xml","image_train"), ("test","test_label.xml","image_test")]
    if (veri_root/"image_query").exists():
        cfgs.append(("query","test_label.xml","image_query"))
    for split, xmlf, imdir in cfgs:
        x, d = veri_root/xmlf, veri_root/imdir
        if not x.exists() or not d.exists():
            continue
        root = load_xml_root_utf8(x)
        for it in root.iter("Item"):
            stats["total"] += 1
            cid, tid = int(it.attrib["colorID"]), int(it.attrib["typeID"])
            if tid not in CAR_TYPES:
                continue
            color_pt = COLOR_MAP.get(VERI_COLOR_ID.get(cid,"unknown"), "outros")
            if color_pt not in ALLOWED_10:
                continue
            stats["cars"] += 1
            stats["colors"][color_pt] += 1
            src = d / it.attrib["imageName"]
            dst_dir = veri_out / split / color_pt
            # cópia segura com retomada
            safe_copy(src, dst_dir, "veri")  # não sobrescreve; cria sufixo se colidir [2]
    return stats  # atende formato de pastas por classe adotado no Ultralytics classify [1]

def process_vcor(vcor_root: Path, out_root: Path) -> dict:
    """Padroniza nomes de cor do VCoR e copia apenas 10 cores para pastas por classe em train/val/test."""
    stats = {"total":0,"colors":defaultdict(int)}
    vcor_out = out_root / "vcor_processed"
    for split in ("train","val","test"):
        sdir = vcor_root/split
        if not sdir.exists():
            continue
        for cdir in sdir.iterdir():
            if not cdir.is_dir():
                continue
            cor = COLOR_MAP.get(cdir.name.lower(), cdir.name.lower())
            if cor not in ALLOWED_10:
                continue
            dst = vcor_out/split/cor
            # inventário existente para retomar
            existing = list_files(dst)
            for img in cdir.glob("*.*"):
                if img.suffix.lower() in (".jpg",".jpeg",".png",".bmp"):
                    stats["total"] += 1
                    stats["colors"][cor] += 1
                    # se já existir um nome igual, safe_copy criará sufixo
                    safe_copy(img, dst, "vcor")  # idempotente em reexecução [2]
    return stats

def merge_10(veri_proc: Path, vcor_proc: Path, out_final: Path) -> dict:
    """Une VeRi (train/test/query->train) + VCoR (train/val/test) nas 10 cores, sem limite e sem sobrescrever."""
    final_stats = defaultdict(lambda: defaultdict(int))
    pools = defaultdict(lambda: defaultdict(list))

    # Mapear VeRi tudo -> train; VCoR: manter splits
    for base, split_map, tag in [
        (veri_proc, {"train":"train","test":"train","query":"train"}, "veri"),
        (vcor_proc, {"train":"train","val":"val","test":"test"}, "vcor")
    ]:
        if not base.exists():
            continue
        for sdir in base.iterdir():
            if not sdir.is_dir() or sdir.name not in split_map:
                continue
            dst_split = split_map[sdir.name]
            for cdir in sdir.iterdir():
                if cdir.is_dir() and cdir.name in ALLOWED_10:
                    pools[dst_split][cdir.name].extend(cdir.glob("*.*"))

    # Copiar todos (sem limite), com verificação de existentes
    for split, cmap in pools.items():
        for color, imgs in cmap.items():
            dst = out_final/split/color
            ensure_dir(dst)
            existing = list_files(dst)
            idx = len(existing)  # continua numeração
            for src in imgs:
                name = f"{color}_{idx:06d}{src.suffix}"
                idx += 1
                dst_path = dst / name
                if dst_path.name in existing:
                    continue  # já existe
                shutil.copy2(src, dst_path)
                final_stats[split][color] += 1

    names = sorted(ALLOWED_10)
    ensure_dir(out_final)
    (out_final/"data.yaml").write_text(
        f"path: {out_final.resolve()}\ntrain: train\nval: val\n"
        + ("test: test\n" if (out_final/'test').exists() else "")
        + f"\nnc: {len(names)}\nnames: {names}\n", encoding="utf-8"
    )  # estrutura esperada pelo yolo classify [1]
    save_json(final_stats, out_final/"dataset_stats.json")
    return {"classes": names, "final_stats": final_stats}

def run(veri_path: str, vcor_path: str, out_path: str):
    veri_root, vcor_root, out_root = Path(veri_path), Path(vcor_path), Path(out_path)
    # Verificações iniciais de existência
    assert veri_root.exists(), f"VeRi não encontrado: {veri_root}"  # checagem básica com Path.exists [2]
    assert vcor_root.exists(), f"VCoR não encontrado: {vcor_root}"  # idem [2]
    ensure_dir(out_root)

    print("-> VeRi"); veri_stats = process_veri(veri_root, out_root)
    print("-> VCoR"); vcor_stats = process_vcor(vcor_root, out_root)

    # Resumos intermediários (inventário)
    save_json(summarize_tree(out_root/"veri_processed"), out_root/"veri_processed_manifest.json")
    save_json(summarize_tree(out_root/"vcor_processed"), out_root/"vcor_processed_manifest.json")

    # Merge final
    final_dir = out_root/"COLOR_FINAL_YOLO"
    info = merge_10(out_root/"veri_processed", out_root/"vcor_processed", final_dir)
    save_json(summarize_tree(final_dir), final_dir/"final_manifest.json")

    print(f"OK VeRi cars: {veri_stats['cars']} | VCoR imgs: {vcor_stats['total']} | Classes: {info['classes']} | Final: {final_dir}")

if __name__ == "__main__":
    run(
        r"C:\Users\riana\OneDrive\Desktop\Vox MVP\data\VeRi",
        r"C:\Users\riana\OneDrive\Desktop\Vox MVP\data\VCor",
        r"C:\Users\riana\OneDrive\Desktop\Vox MVP\data\PROCESSED"
    )
