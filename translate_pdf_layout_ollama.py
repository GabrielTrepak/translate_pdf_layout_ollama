# translate_pdf_layout_ollama_v4.py
# EN -> PT-BR (light novel) mantendo layout (bbox) do PDF, usando Ollama local
# v4 + fix de caracteres que viram "?" no PDF (ex.: reticências … -> ...)
#
# Principais features:
# - Character Bible (CHAR_DB) com gênero + registro
# - SPEAKER_STATE = personagem atual (persistente)
# - Prompts com registro por personagem
# - 2-pass: translate + polish
# - Guardrails: evita "você" fora de aspas; corrige gênero 1ª pessoa; remove honoríficos inventados
# - Layout: MIN_FONT_SIZE=8.0, lineheight (fallback)
# - Se não couber no mínimo: truncamento com "..." (não usa "…")
# - Sanitização: normaliza caracteres para evitar glifos faltando (que viram "?")

import os
import re
import json
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

import fitz  # PyMuPDF
import requests


# =========================
# CONFIG
# =========================

PDF_DIR = r"C:\Users\gtrep\Desktop\Python"
PDF_IN = os.path.join(PDF_DIR, "Tsuki ga Michibiku Isekai Douchuu_01-1-9.pdf")
PDF_OUT = os.path.join(PDF_DIR, "Tsuki_traduzido_layout_v4_fixed.pdf")

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"

CACHE_FILE = os.path.join(PDF_DIR, "translate_cache_en_pt.json")

MAX_CHARS_PER_CALL = 1400

# Layout / legibilidade
MIN_FONT_SIZE = 10.0
FONT_STEP = 0.5
LINEHEIGHT = 1.15  # se sua versão do PyMuPDF aceitar

REDACTION_FILL = (1, 1, 1)
TEXT_COLOR = (0, 0, 0)
TEXT_ALIGN = 0  # left


# =========================
# Character Bible
# =========================

CHAR_DB: Dict[str, Dict[str, str]] = {
    "makoto":    {"gender": "male",   "register": "informal_neutro"},
    "tsukuyomi": {"gender": "male",   "register": "formal_moderado"},
    "goddess":   {"gender": "female", "register": "formal_frio"},
    "tomoe":     {"gender": "female", "register": "informal_brincalhona"},
    "mio":       {"gender": "female", "register": "emocional_obsessiva"},
    "shiki":     {"gender": "male",   "register": "formal_calmo"},
    "tamaki":    {"gender": "female", "register": "neutro"},
    "emma":      {"gender": "female", "register": "neutro"},
    "lime":      {"gender": "male",   "register": "neutro"},
    "root":      {"gender": "unknown","register": "neutro"},
    "sofia":     {"gender": "female", "register": "neutro"},
    "hibiki":    {"gender": "female", "register": "neutro"},
}

NAME_ALIASES: Dict[str, List[str]] = {
    "makoto":    ["makoto", "misumi", "misumi makoto"],
    "tsukuyomi": ["tsukuyomi"],
    "goddess":   ["goddess", "the goddess"],
    "tomoe":     ["tomoe", "shin"],
    "mio":       ["mio"],
    "shiki":     ["shiki", "lich"],
    "tamaki":    ["tamaki"],
    "emma":      ["emma"],
    "lime":      ["lime", "lime latte"],
    "root":      ["root"],
    "sofia":     ["sofia", "sofia bulga"],
    "hibiki":    ["hibiki", "hibiki otonashi"],
}

STYLE_BIBLE = (
    "GUIA DE ESTILO (obrigatório):\n"
    "- Tom geral: PT-BR natural de light novel.\n"
    "- Evite formalidade do nada (ex.: 'Vossa/Sua Alteza'), a menos que haja reverência explícita no EN.\n"
    "- Palavrões: só se existirem no EN; caso contrário, suavize.\n"
    "- Mantenha consistência de registro dentro do mesmo diálogo.\n"
    "- Preserve pontuação e quebras de linha.\n"
)

CONTEXT_WINDOW = 3
context_memory = deque(maxlen=CONTEXT_WINDOW)

# speaker atual persistente
SPEAKER_STATE = "makoto"


# =========================
# TEXT SANITIZATION (evita "Fo?" etc.)
# =========================

def sanitize_pdf_text(s: str) -> str:
    if not s:
        return ""

    # normaliza caracteres que frequentemente viram "?" por falta de glifo na fonte
    s = (s
         .replace("…", "...")
         .replace("—", "-")
         .replace("–", "-")
         .replace("“", '"')
         .replace("”", '"')
         .replace("‘", "'")
         .replace("’", "'")
    )

    # remove caracteres invisíveis/controle (mantém \n e \t)
    s = "".join(
        ch for ch in s
        if ch == "\n" or ch == "\t" or (ord(ch) >= 32 and ord(ch) != 127)
    )

    # limpa espaços duplicados
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


# =========================
# CACHE
# =========================

def load_cache() -> Dict[str, str]:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            obj = json.load(f)
            return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def save_cache(cache: Dict[str, str]) -> None:
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def key_for(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# =========================
# OLLAMA
# =========================

def ollama_generate(prompt: str, timeout_sec: int = 180) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_ctx": 4096,
        },
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout_sec)
    r.raise_for_status()
    return (r.json().get("response", "") or "").strip()


# =========================
# Speaker detection
# =========================

def detect_speaker(en_text: str) -> str:
    global SPEAKER_STATE
    t = (en_text or "").lower()

    for key, aliases in NAME_ALIASES.items():
        for a in aliases:
            if a in t:
                return key

    return SPEAKER_STATE


# =========================
# Prompts
# =========================

def speaker_rules(speaker: str) -> str:
    meta = CHAR_DB.get(speaker, {"gender": "unknown", "register": "neutro"})
    gender = meta.get("gender", "unknown")
    register = meta.get("register", "neutro")

    if speaker == "makoto":
        return (
            "- Speaker dominante: Makoto (narrador, homem).\n"
            "- Fora de aspas: 1ª pessoa masculina, registro informal neutro.\n"
            "- Não transformar 'eu' em 'você'.\n"
        )

    if speaker == "tsukuyomi":
        return (
            "- Speaker dominante: Tsukuyomi (homem).\n"
            "- Diálogos dele: formal moderado, calmo; sem gíria.\n"
        )

    if speaker == "goddess":
        return (
            "- Speaker dominante: Deusa do mundo (mulher).\n"
            "- Diálogos dela: formal/frio; pode soar arrogante.\n"
        )

    return (
        f"- Speaker dominante provável: {speaker}.\n"
        f"- Gênero: {gender}. Registro: {register}.\n"
        "- Preserve consistência nesse registro.\n"
    )

def build_translate_prompt(text: str, context: str, speaker: str) -> str:
    return (
        "Você é um tradutor profissional de light novel.\n"
        "Traduza do INGLÊS para PORTUGUÊS DO BRASIL com linguagem NATURAL.\n\n"
        f"{STYLE_BIBLE}\n"
        "REGRAS DURAS:\n"
        "- Texto entre aspas (\" \") é DIÁLOGO. Fora de aspas é NARRAÇÃO/pensamento.\n"
        "- Narrador padrão: Makoto (HOMEM) -> fora de aspas, 1ª pessoa masculina.\n"
        "- Não invente honoríficos ('Vossa/Sua Alteza') sem reverência explícita no EN.\n"
        "- Retorne APENAS a tradução.\n\n"
        "REGRAS DE PERSONAGEM:\n"
        f"{speaker_rules(speaker)}\n"
        "Contexto anterior (referência):\n"
        f"{context}\n\n"
        "Texto:\n"
        f"{text}\n"
    )

def build_polish_prompt(original_en: str, draft_pt: str, context: str, speaker: str) -> str:
    return (
        "Você é um revisor de tradução PT-BR (light novel).\n"
        "Melhore naturalidade e consistência de registro, sem mudar o sentido.\n\n"
        f"{STYLE_BIBLE}\n"
        "REGRAS DURAS:\n"
        "- Fora de aspas: mantenha o narrador (Makoto) em 1ª pessoa masculina quando aplicável.\n"
        "- Se aparecer 'você' fora de aspas, corrija para 1ª pessoa.\n"
        "- Remova honoríficos inventados se não existirem no EN.\n"
        "- Corrija concordância e termos estranhos.\n"
        "- Retorne APENAS o texto revisado.\n\n"
        "REGRAS DE PERSONAGEM:\n"
        f"{speaker_rules(speaker)}\n"
        "Contexto anterior:\n"
        f"{context}\n\n"
        "Original (EN):\n"
        f"{original_en}\n\n"
        "Rascunho (PT):\n"
        f"{draft_pt}\n"
    )


# =========================
# Guardrails
# =========================

def has_quotes(s: str) -> bool:
    return '"' in (s or "")

def contains_first_person(s: str) -> bool:
    return bool(re.search(r"\b(eu|meu|minha|mim|comigo|me)\b", (s or "").lower()))

def looks_like_second_person_leak(pt: str) -> bool:
    t = (pt or "").lower()
    return ("você" in t) and (not has_quotes(pt))

def looks_like_wrong_gender_first_person(pt: str) -> bool:
    t = (pt or "").lower()
    if not contains_first_person(t):
        return False
    bad = [
        r"\beu\s+(?:estava|fiquei|continuei|pareci|me\s+senti)\s+\w+?a\b",
        r"\beu\s+sou\s+\w+?a\b",
        r"\bestou\s+\w+?a\b",
    ]
    return any(re.search(p, t) for p in bad)

def contains_reverence_en(en: str) -> bool:
    t = (en or "").lower()
    return any(x in t for x in [
        "your highness", "his highness", "her highness",
        "your majesty", "lord", "my lord", "lady", "my lady"
    ])

def strip_unjustified_honorifics(pt: str, en: str) -> str:
    if contains_reverence_en(en):
        return pt
    out = pt
    out = re.sub(r"\b(Sua|Vossa)\s+Alteza\b\s*", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\b(Sua|Vossa)\s+Majestade\b\s*", "", out, flags=re.IGNORECASE)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()


# =========================
# Chunking / Translation
# =========================

def split_into_paragraphish_chunks(text: str, max_chars: int) -> List[str]:
    raw_lines = (text or "").splitlines()
    paragraphs: List[str] = []
    buf: List[str] = []

    def flush():
        if buf:
            paragraphs.append("\n".join(buf).strip())
            buf.clear()

    for line in raw_lines:
        if line.strip() == "":
            flush()
        else:
            buf.append(line.rstrip())
    flush()

    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: List[str] = []
    cur = ""
    for para in paragraphs:
        if not para:
            continue

        if len(para) > max_chars:
            sub = ""
            for ln in para.splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                if len(sub) + len(ln) + 1 > max_chars:
                    if sub:
                        chunks.append(sub)
                    sub = ln
                else:
                    sub = f"{sub}\n{ln}" if sub else ln
            if sub:
                chunks.append(sub)
            continue

        if len(cur) + len(para) + 2 > max_chars:
            if cur:
                chunks.append(cur)
            cur = para
        else:
            cur = f"{cur}\n\n{para}" if cur else para

    if cur:
        chunks.append(cur)

    return chunks


def translate_then_polish(en_text: str, cache: Dict[str, str]) -> str:
    global SPEAKER_STATE

    t = (en_text or "").strip()
    if not t:
        return ""

    k = key_for("V4FIX|" + t)
    if k in cache:
        return cache[k]

    speaker = detect_speaker(t)
    SPEAKER_STATE = speaker

    context = "\n\n".join(context_memory)

    prompt = build_translate_prompt(t, context, speaker)
    draft = ollama_generate(prompt)

    if looks_like_second_person_leak(draft) or looks_like_wrong_gender_first_person(draft):
        retry = (
            prompt
            + "\n\nRefaça obedecendo estritamente:\n"
              "- Fora de aspas: narrador em 1ª pessoa masculina quando aplicável.\n"
              "- NÃO use 'você' fora de aspas.\n"
              "- Não invente honoríficos.\n"
              "Retorne só a tradução.\n"
        )
        draft2 = ollama_generate(retry)
        if draft2.strip():
            draft = draft2.strip()

    draft = strip_unjustified_honorifics(draft, t)

    polish_prompt = build_polish_prompt(t, draft, context, speaker)
    polished = ollama_generate(polish_prompt)

    out = polished.strip() if polished.strip() else draft.strip()
    out = strip_unjustified_honorifics(out, t)

    # sanitiza caracteres problemáticos para fonte do PDF
    out = sanitize_pdf_text(out)

    if out and (not has_quotes(t)):
        context_memory.append(out)

    cache[k] = out
    return out


def translate_block(text: str, cache: Dict[str, str]) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    chunks = split_into_paragraphish_chunks(t, MAX_CHARS_PER_CALL)
    out_parts: List[str] = []
    for ch in chunks:
        out_parts.append(translate_then_polish(ch, cache))
    return "\n\n".join([x for x in out_parts if x is not None]).strip()


# =========================
# PDF: collect blocks
# =========================

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


# =========================
# PDF: writing (fit + truncation) + sanitize
# =========================

def truncate_to_fit(page: Any, bbox: fitz.Rect, text: str, font: str, fontsize: float) -> str:
    s = sanitize_pdf_text((text or "").strip())
    if not s:
        return s

    # teste rápido: já cabe?
    shape = page.new_shape()
    try:
        rc = shape.insert_textbox(
            bbox, s, fontname=font, fontsize=fontsize, align=TEXT_ALIGN, color=TEXT_COLOR, lineheight=LINEHEIGHT
        )
    except TypeError:
        rc = shape.insert_textbox(
            bbox, s, fontname=font, fontsize=fontsize, align=TEXT_ALIGN, color=TEXT_COLOR
        )
    if isinstance(rc, (int, float)) and rc >= 0:
        return s

    # trunca sem cortar feio (corta até o último espaço)
    min_len = max(20, int(len(s) * 0.3))
    cur = s

    while len(cur) > min_len:
        # corta um bloco e depois volta ao último espaço
        cur = cur[:-30].rstrip()
        if " " in cur:
            cur = cur.rsplit(" ", 1)[0].rstrip()

        cur_try = cur + "..."
        shape2 = page.new_shape()
        try:
            rc2 = shape2.insert_textbox(
                bbox, cur_try, fontname=font, fontsize=fontsize, align=TEXT_ALIGN, color=TEXT_COLOR, lineheight=LINEHEIGHT
            )
        except TypeError:
            rc2 = shape2.insert_textbox(
                bbox, cur_try, fontname=font, fontsize=fontsize, align=TEXT_ALIGN, color=TEXT_COLOR
            )

        if isinstance(rc2, (int, float)) and rc2 >= 0:
            return cur_try

    first_line = s.splitlines()[0].strip()
    return (first_line[: max(20, min(len(first_line), 120))] + "...") if first_line else "..."


def fit_textbox(page: Any, bbox: fitz.Rect, text: str, font: str, start_size: float) -> bool:
    # clamp: se o original já era pequeno, não deixa ficar menor que o mínimo
    cur = max(float(start_size), MIN_FONT_SIZE)
    txt = sanitize_pdf_text(text)

    while cur >= MIN_FONT_SIZE:
        shape = page.new_shape()
        try:
            rc = shape.insert_textbox(
                bbox,
                txt,
                fontname=font,
                fontsize=cur,
                align=TEXT_ALIGN,
                color=TEXT_COLOR,
                lineheight=LINEHEIGHT,
            )
        except TypeError:
            rc = shape.insert_textbox(
                bbox,
                txt,
                fontname=font,
                fontsize=cur,
                align=TEXT_ALIGN,
                color=TEXT_COLOR,
            )

        if isinstance(rc, (int, float)) and rc < 0:
            cur -= FONT_STEP
            continue

        shape.commit()
        return True

    # Não coube: trunca em vez de encolher
    fitted = truncate_to_fit(page, bbox, txt, font, MIN_FONT_SIZE)
    shape = page.new_shape()
    try:
        shape.insert_textbox(
            bbox, fitted, fontname=font, fontsize=MIN_FONT_SIZE, align=TEXT_ALIGN, color=TEXT_COLOR, lineheight=LINEHEIGHT
        )
    except TypeError:
        shape.insert_textbox(
            bbox, fitted, fontname=font, fontsize=MIN_FONT_SIZE, align=TEXT_ALIGN, color=TEXT_COLOR
        )
    shape.commit()
    return False


# =========================
# MAIN
# =========================

def main() -> None:
    if not os.path.exists(PDF_IN):
        raise FileNotFoundError(f"PDF não encontrado em: {PDF_IN}")

    # sanity: Ollama online
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

        translated_items: List[Tuple[fitz.Rect, str, str, float]] = []
        for c in candidates:
            tr = translate_block(c["text"], cache)
            if not tr.strip():
                skipped_empty += 1
                continue
            translated_items.append((c["bbox"], tr, c["font"], c["size"]))

        if not translated_items:
            continue

        translated_pages += 1

        for bbox, _, _, _ in translated_items:
            page.add_redact_annot(bbox, fill=REDACTION_FILL)
        page.apply_redactions()

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
    print(f"Paginas com traducao: {translated_pages}")
    print(f"Blocos traduzidos: {translated_blocks}")
    print(f"Blocos pulados: {skipped_empty}")
    print(f"Saida: {PDF_OUT}")
    print(f"Cache: {CACHE_FILE}\n")


if __name__ == "__main__":
    main()
