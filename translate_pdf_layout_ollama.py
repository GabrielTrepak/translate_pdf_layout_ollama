import os
import re
import json
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import requests


PDF_DIR = r"C:\Users\gtrep\Desktop\Python"
PDF_IN = os.path.join(PDF_DIR, "Tsuki ga Michibiku Isekai Douchuu_01-1-9.pdf")
PDF_OUT = os.path.join(PDF_DIR, "Tsuki_traduzido_layout.pdf")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"

CACHE_FILE = os.path.join(PDF_DIR, "translate_cache_en_pt.json")

MAX_CHARS_PER_CALL = 1400
MIN_FONT_SIZE = 6.0
FONT_STEP = 0.5

REDACTION_FILL = (1, 1, 1)
TEXT_COLOR = (0, 0, 0)  # GARANTE preto


def load_cache() -> Dict[str, str]:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_cache(cache: Dict[str, str]) -> None:
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def key_for(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ollama_translate(text: str, timeout_sec: int = 180) -> str:
    prompt = (
        "Você é um tradutor profissional.\n"
        "Traduza do INGLÊS para PORTUGUÊS DO BRASIL.\n"
        "Regras:\n"
        "- Mantenha o sentido, nomes próprios e números.\n"
        "- Preserve a pontuação.\n"
        "- Se houver quebras de linha, mantenha quebras equivalentes.\n"
        "- Não explique nada. Retorne APENAS a tradução.\n\n"
        "Texto:\n"
        f"{text}"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout_sec)
    r.raise_for_status()
    return (r.json().get("response", "") or "").strip()


def translate_block(text: str, cache: Dict[str, str]) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    def translate_piece(piece: str) -> str:
        k = key_for(piece)
        if k in cache:
            return cache[k]
        tr = ollama_translate(piece)
        cache[k] = tr
        return tr

    if len(t) <= MAX_CHARS_PER_CALL:
        return translate_piece(t)

    parts: List[str] = []
    cur = ""
    for line in t.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(cur) + len(line) + 1 > MAX_CHARS_PER_CALL:
            if cur:
                parts.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        parts.append(cur)

    out = [translate_piece(p) for p in parts]
    return "\n".join(out).strip()


def is_noise(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if len(t) <= 2:
        return True
    if all(ch.isdigit() or ch in ".,:/-–—()[]{}<>|•·* \n\t" for ch in t):
        return True
    if re.fullmatch(r"\d{1,3}", t):
        return True
    return False


def collect_text_blocks(page: Any) -> List[Dict[str, Any]]:
    d: Dict[str, Any] = page.get_text("dict")
    blocks: List[Dict[str, Any]] = d.get("blocks", [])
    candidates: List[Dict[str, Any]] = []

    for b in blocks:
        if b.get("type") != 0:
            continue

        bbox = fitz.Rect(b["bbox"])
        lines: List[Dict[str, Any]] = b.get("lines", [])
        if not lines:
            continue

        text_lines: List[str] = []
        base_size: Optional[float] = None

        for ln in lines:
            spans: List[Dict[str, Any]] = ln.get("spans", [])
            if not spans:
                continue

            if base_size is None:
                base_size = float(spans[0].get("size", 10.0))

            s = "".join(str(sp.get("text", "")) for sp in spans).rstrip()
            if s:
                text_lines.append(s)

        text = "\n".join(text_lines).strip()
        if is_noise(text):
            continue

        candidates.append({
            "bbox": bbox,
            "text": text,
            "font": "helv",
            "size": base_size or 10.0,
        })

    return candidates


def clear_box(page: Any, bbox: fitz.Rect) -> None:
    page.draw_rect(bbox, color=None, fill=REDACTION_FILL)


def fit_textbox(page: Any, bbox: fitz.Rect, text: str, font: str, start_size: float) -> bool:
    """
    PyMuPDF: insert_textbox retorna:
      rc > 0  -> cabem todos os caracteres (rc = espaço vertical sobrando)
      rc < 0  -> overflow (não coube)
    Para não “sujar” a página a cada tentativa, usamos Shape e só commit quando couber.
    """
    cur = float(start_size)

    while cur >= MIN_FONT_SIZE:
        shape = page.new_shape()
        rc = shape.insert_textbox(
            bbox,
            text,
            fontname=font,
            fontsize=cur,
            align=0,
            color=TEXT_COLOR,  # preto
        )

        if isinstance(rc, (int, float)) and rc < 0:
            # NÃO coube: descarta e tenta menor
            cur -= FONT_STEP
            continue

        # Coube: aplica no PDF de verdade
        shape.commit()
        return True

    # Se nunca coube, escreve no mínimo mesmo assim
    shape = page.new_shape()
    shape.insert_textbox(
        bbox,
        text,
        fontname=font,
        fontsize=MIN_FONT_SIZE,
        align=0,
        color=TEXT_COLOR,
    )
    shape.commit()
    return False


def main() -> None:
    if not os.path.exists(PDF_IN):
        raise FileNotFoundError(f"PDF não encontrado em: {PDF_IN}")

    # sanity check: ollama
    requests.get("http://localhost:11434", timeout=3)

    cache = load_cache()
    doc: Any = fitz.open(PDF_IN)

    translated_pages = 0
    translated_blocks = 0
    skipped_empty = 0

    for i in range(int(doc.page_count)):
        page: Any = doc.load_page(i)
        candidates = collect_text_blocks(page)
        if not candidates:
            continue

        # 1) traduz primeiro (sem apagar ainda)
        translated_items: List[Tuple[fitz.Rect, str, str, float]] = []
        for c in candidates:
            try:
                tr = translate_block(c["text"], cache)
            except Exception as e:
                tr = ""
            if not tr.strip():
                skipped_empty += 1
                continue
            translated_items.append((c["bbox"], tr, c["font"], c["size"]))

        if not translated_items:
            continue

        translated_pages += 1

        # 2) redige só o que tem tradução
        for bbox, _, _, _ in translated_items:
            page.add_redact_annot(bbox, fill=REDACTION_FILL)
        page.apply_redactions()

        # 3) escreve tradução
        for bbox, tr, font, size in translated_items:
            fit_textbox(page, bbox, tr, font, size)
            translated_blocks += 1

        print(f"[OK] Página {i + 1}: {len(translated_items)} blocos traduzidos (pulados vazios: {skipped_empty})")

        if (i + 1) % 2 == 0:
            save_cache(cache)

    doc.save(PDF_OUT, deflate=True)
    doc.close()
    save_cache(cache)

    print("\nFINALIZADO")
    print(f"Páginas com tradução: {translated_pages}")
    print(f"Blocos traduzidos: {translated_blocks}")
    print(f"Blocos pulados (tradução vazia/erro): {skipped_empty}")
    print(f"Saída: {PDF_OUT}")
    print(f"Cache: {CACHE_FILE}\n")


if __name__ == "__main__":
    main()
