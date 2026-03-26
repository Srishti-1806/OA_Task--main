import re

UNITS = {
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4,
    "पांच": 5, "पाँच": 5, "छह": 6, "छः": 6, "सात": 7,
    "आठ": 8, "नौ": 9,
}

TEENS = {
    "दस": 10, "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14,
    "पंद्रह": 15, "सोलह": 16, "सत्रह": 17, "अठारह": 18, "उन्नीस": 19,
}

TENS = {
    "बीस": 20, "इक्कीस": 21, "बाइस": 22, "तेइस": 23, "चौबीस": 24,
    "पच्चीस": 25, "छब्बीस": 26, "सत्ताइस": 27, "अट्ठाइस": 28, "उनतीस": 29,
    "तीस": 30, "इकतीस": 31, "बत्तीस": 32, "तैंतीस": 33, "चौंतीस": 34,
    "पैंतीस": 35, "छत्तीस": 36, "सैंतीस": 37, "अड़तीस": 38, "उनतालीस": 39,
    "चालीस": 40, "इकतालीस": 41, "बयालीस": 42, "तैंतालीस": 43, "चवालीस": 44,
    "पैंतालीस": 45, "छियालीस": 46, "सैंतालीस": 47, "अड़तालीस": 48, "उनचास": 49,
    "पचास": 50, "इक्यावन": 51, "बावन": 52, "तिरपन": 53, "चौवन": 54,
    "पचपन": 55, "छप्पन": 56, "सत्तावन": 57, "अट्ठावन": 58, "उनसठ": 59,
    "साठ": 60, "इकसठ": 61, "बासठ": 62, "तिरसठ": 63, "चौंसठ": 64,
    "पैंसठ": 65, "छियासठ": 66, "सड़सठ": 67, "अड़सठ": 68, "उनहत्तर": 69,
    "सत्तर": 70, "इकहत्तर": 71, "बहत्तर": 72, "तिहत्तर": 73, "चौहत्तर": 74,
    "पचहत्तर": 75, "छिहत्तर": 76, "सतहत्तर": 77, "अठहत्तर": 78, "उनासी": 79,
    "अस्सी": 80, "इक्यासी": 81, "बयासी": 82, "तिरासी": 83, "चौरासी": 84,
    "पचासी": 85, "छियासी": 86, "सतासी": 87, "अठासी": 88, "नवासी": 89,
    "नब्बे": 90, "इक्यानवे": 91, "बानवे": 92, "तिरानवे": 93, "चौरानवे": 94,
    "पचानवे": 95, "छियानवे": 96, "सत्तानवे": 97, "अट्ठानवे": 98, "निन्यानवे": 99,
}

MULTIPLIERS = {
    "सौ": 100, "हजार": 1000, "हज़ार": 1000,
    "लाख": 100000, "करोड़": 10000000,
}

ALL_NUM_WORDS = {**UNITS, **TEENS, **TENS, **MULTIPLIERS}

SKIP_PATTERNS = [
    r"दो-चार", r"चार-पांच", r"दो-एक", r"एक-दो",
    r"दो-तीन", r"तीन-चार", r"पांच-सात", r"आठ-दस",
    r"एक-आध", r"सौ-दो\s*सौ",
]

IDIOMATIC_PHRASES = [
    "दो-चार बातें", "दो-चार दिन", "एक-आध",
    "चार-पांच बातें", "एक दो बार", "दो चार बार",
    "एक न एक", "दो-दो", "तीन-तीन", "चार-चार",
]

VERB_CONTEXT_BEFORE = {"दे", "कर", "बता", "ला", "भेज", "सुना", "दिखा", "बना", "रख", "हटा"}


def is_idiomatic(text, start_idx, end_idx, words):
    context = " ".join(words[max(0, start_idx-1):min(len(words), end_idx+2)])
    for pattern in SKIP_PATTERNS:
        if re.search(pattern, context):
            return True
    for phrase in IDIOMATIC_PHRASES:
        if phrase in context:
            return True
    return False


def parse_compound_number(num_words):
    total = 0
    current = 0
    hundreds = 0

    for w in num_words:
        if w in UNITS:
            current = UNITS[w]
        elif w in TEENS or w in TENS:
            current = ALL_NUM_WORDS[w]
        elif w in MULTIPLIERS:
            mult = MULTIPLIERS[w]
            if current == 0:
                current = 1
            if mult == 100:
                hundreds = current * mult
                current = 0
            else:
                total += (hundreds + current) * mult
                hundreds = 0
                current = 0
        else:
            return None

    total += hundreds + current
    return total if total > 0 else None


def normalize_numbers(text):
    words = text.split()
    result = []
    i = 0

    while i < len(words):
        if re.search(r"[।\-–]", words[i]) and any(nw in words[i] for nw in ALL_NUM_WORDS):
            result.append(words[i])
            i += 1
            continue

        if words[i] in ALL_NUM_WORDS:
            if words[i] == "दो" and i > 0 and words[i-1] in VERB_CONTEXT_BEFORE:
                result.append(words[i])
                i += 1
                continue

            num_words = []
            start_idx = i

            while i < len(words) and words[i] in ALL_NUM_WORDS:
                num_words.append(words[i])
                i += 1

            if is_idiomatic(text, start_idx, i, words):
                result.extend(num_words)
                continue

            value = parse_compound_number(num_words)
            if value is not None:
                result.append(str(value))
            else:
                result.extend(num_words)
        else:
            result.append(words[i])
            i += 1

    return " ".join(result)


def demo():
    examples = [
        ("उसने दो किताबें खरीदीं", "Simple"),
        ("दस लोग आए थे", "Simple"),
        ("सौ रुपये दे दो", "Simple"),
        ("मेरे पास पच्चीस रुपये हैं", "Compound"),
        ("तीन सौ चौवन रुपये लगे", "Compound"),
        ("एक हज़ार रुपये चाहिए", "Compound"),
        ("दो-चार बातें करनी हैं", "Idiomatic"),
        ("एक-आध बार और आओ", "Idiomatic"),
        ("हम दो-तीन दिन रुकेंगे", "Idiomatic"),
    ]

    print("NUMBER NORMALIZATION — Before/After Examples")
    print("=" * 70)
    for text, desc in examples:
        result = normalize_numbers(text)
        changed = "✓" if result != text else "✗ (kept as-is)"
        print(f"\n  [{desc}]")
        print(f"  Before: {text}")
        print(f"  After:  {result} {changed}")


if __name__ == "__main__":
    demo()
