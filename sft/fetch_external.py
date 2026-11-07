import json
import random
import re
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

path = Path(__file__).parent
raw = path / "hug_dataset" / "raw"

MAX_PER_SOURCE = 5000
Q_CHARS = (5, 4200)
A_CHARS = (3, 4602)
MAX_SENTENCES = 300
rng = random.Random(42)

ASKS_EN = (
    r"^(what|how|why|when|where|which|who|whom|whose|explan|describe|define|list|name|give|tell|"
    r"compare|summarize|translate|help|can you|could you|please|is|are|does|do|did|should|would|"
    r"write|draft|create|make|suggest|recommend|find|plan)\b"   # amaliy iltimoslar: xat, reja, ro'yxat
    r"|\?\s*$"                                    # oxiri "?" bilan tugasa
)
ASKS_UZ = (
    r"\?\s*$"                                     # oxiri "?" bilan tugasa
    r"|\b(nima|nimaga|nega|qanday|qanaqa|qaysi|qachon|qayer|qayerda|qayerga|qayerdan|"
    r"kim|kimga|nechta|necha|qancha|qanchalik)"   # so'roq so'zlar (qo'shimcha olishi mumkin: nimalar, kimning)
    r"|\b\w+mi\b"                                 # -mi yuklamasi: kerakmi, bormi, mumkinmi
    r"|tushuntir|ta'rifla|izohla|farqi|sanab|nomla|aytib|ayting|maslahat|yordam ber|"
    r"tarjima qil|misol keltir|qisqacha|qisqartir|xulosa qil|solishtir|yozib ber|yozing|tuzib ber|"
    r"tavsiya qil|reja tuz"
)

ASKS_CY = (
    r"\?\с*$"                                     # охири "?" билан тугаса
    r"|\б(нима|нимага|нега|қандай|қанақа|қайси|қачон|қаер|қаерда|қаерга|қаердан|"
    r"ким|кимга|нечта|неча|қанча|қанчалик)"   # сўроқ сўзлар (қўшимча олиши мумкин: нималар, кимнинг)
    r"|\б\w+ми\б"                                 # -ми юкламаси: керакми, борми, мумкинми
    r"|тушунтир|таърифла|изоҳла|фарқи|санаб|номла|айтиб|айтинг|маслаҳат|ёрдам бер|"
    r"таржима қил|мисол келтир|қисқача|қисқартир|хулоса қил|солиштир|ёзиб бер|ёзинг|тузиб бер|"
    r"тавсия қил|режа туз"
)

# Sifat filtri: havola, kod, markdown va modelning o'zi haqidagi gaplar spec'dagi uslubga to'g'ri kelmaydi.
BAD_ANSWER = r"```|`|https?://|www\.|\*\*|^#|^\s*[-*>] |^ {4}|" \
             r"\b(I am|I'm|I was|I cannot|I can't|I don't|my training|as an AI|as a language model)\b|" \
             r"LAION|Open.?Assistant|sun'iy intellekt sifatida|til modeli sifatida|AI sifatida|men sun'iy"


CREATIVE = r"\b(write|generate|create|compose|rewrite|pretend|imagine|recommend|suggest|you are)\b|" \
           r"yarating|yarat\b|yozing|yozib|taklif qil|sarlavha|tvit|she'r|hikoya|shior|qayta yoz|tasavvur qil|" \
           r"\b(latest|newest|current|do you think|do you know|companies)\b|eng yangi|^[\"“”«']"

CYRILLIC = re.compile(r"[Ѐ-ӿ]")

def plan(text):
    return re.sub(r"[‘’ʻʼ]", "'", text)

def fits(q, a):
    q, a = q.strip(), a.strip()

    if "\n" in q or not Q_CHARS[0] <= len(q) <=  Q_CHARS[1] or not A_CHARS[0] <= len(a) <= A_CHARS[1]:
        return False
    if re.search(BAD_ANSWER, plan(a), flags=re.I | re.M) or re.search(r"^(Q: |## )", a, flags=re.M):
        return False
    if re.search(CREATIVE, plan(q), flags=re.I):
        return False
    
    return len(re.findall(r"[.!?](\s|$)", a)) <= MAX_SENTENCES

def uz_apostrophes(text):
    """Tarjimadagi to'g'ri apostrofni spec'dagi shaklga keltiradi: o' g' -> o‘ g‘, qolgani -> ’."""
    text = re.sub(r"([oOgG])['`ʻʼ‘’]", "\\1‘", text)
    return re.sub(r"(?<=\w)['`ʻʼ](?=\w)", "’", text)



def gen_dataset():
    """Birinchi savol + eng yuqori baholangan javob (rank 0), faqat ingliz tili va tekshiruvdan o'tganlari."""
    path = hf_hub_download("MLDataScientist/SFT-UZ-9k", "output_uz_sft.json",
                           repo_type="dataset", local_dir=raw / "sft_uz")

    rows = json.loads(Path(path).read_text(encoding="utf-8"))

    stats = {"savol-javob juftligi": 0}
    out = []

    for r in rows: 
        q, a = r.get("prompt") or "", r.get("answer") or ""
        
        stats["savol-javob juftligi"] += 1
        if not fits(q, a):
            continue
        out.append((uz_apostrophes(q.strip()), uz_apostrophes(a.strip())))

    return out, stats

def write(name, header, pairs):
    pairs = sorted(set(pairs))
    rng.shuffle(pairs)
    pairs = pairs[:MAX_PER_SOURCE]
    body = "".join(f"Q: {q}\nA: {a}\n\n" for q, a in pairs)
    (path / name).write_text(f"{header}\n## external\n\n{body}", encoding="utf-8")
    return len(pairs)

def main():
    for name, load, header in (
        ("sft_uz.txt", gen_dataset,
         "# Manba: https://huggingface.co/datasets/MLDataScientist/SFT-UZ-9k, litsenziya Apache License 2.0.\n"
         "# Alpaca datasetining o'zbek tiliga mashina tarjimasi."),
    ):
        pairs, stats = load()
        kept = write(name, header, pairs)
        print(name)
        for step, n in stats.items():
            print(f"  {step:<22} {n}")
        print(f"  {'faylga yozildi':<22} {kept}")


if __name__ == "__main__":
    main()