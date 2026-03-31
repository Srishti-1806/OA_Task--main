import re
import unicodedata


def is_latin_script(word):
    """Check if word contains Latin characters."""
    return bool(re.match(r'^[a-zA-Z]+$', word))


def is_devanagari(word):
    """Check if word is primarily Devanagari."""
    devanagari_count = sum(1 for c in word if '\u0900' <= c <= '\u097F')
    return devanagari_count > len(word) * 0.5


ENGLISH_LOANWORDS = {
    "कंप्यूटर", "कम्प्यूटर", "लैपटॉप", "मोबाइल", "फ़ोन", "फोन",
    "इंटरनेट", "ऑनलाइन", "ऑफलाइन", "वाईफ़ाई", "वेबसाइट",
    "सॉफ्टवेयर", "हार्डवेयर", "ऐप", "एप्लीकेशन",
    "इंटरव्यू", "जॉब", "करियर", "कैरियर", "ऑफिस", "कंपनी",
    "बिज़नेस", "बिजनेस", "मैनेजर", "बॉस", "सैलरी", "प्रमोशन",
    "कॉलेज", "यूनिवर्सिटी", "स्कूल", "क्लास", "टीचर",
    "स्टूडेंट", "डिग्री", "सर्टिफिकेट", "एग्जाम",
    "प्रॉब्लम", "प्रोब्लम", "सॉल्व", "सोल्व", "टाइम", "टाईम",
    "फैमिली", "फ्रेंड", "फ्रेंड्स", "पार्टी", "ट्रेन", "बस",
    "टिकट", "होटल", "रेस्टोरेंट", "शॉपिंग", "मार्केट",
    "हॉस्पिटल", "डॉक्टर", "नर्स", "मेडिसिन",
    "फेसबुक", "इंस्टाग्राम", "ट्विटर", "यूट्यूब", "वीडियो",
    "फ़ोटो", "फोटो", "सेल्फी", "ब्लॉग", "पोस्ट", "लाइक",
    "कमेंट", "शेयर", "फॉलो", "सब्सक्राइब",
    "लाइफ", "लाइफस्टाइल", "डाइट", "फिटनेस", "जिम", "योगा",
    "स्ट्रेस", "डिप्रेशन", "मोटिवेशन", "इंस्पिरेशन",
    "टॉपिक", "प्रोग्राम", "प्रोजेक्ट", "रिपोर्ट", "प्लान",
    "ट्राई", "मैनेज", "हैंडल", "फोकस", "अचीव",
    "सिंपल", "इजी", "डिफिकल्ट", "इम्पॉर्टेंट", "स्पेशल",
    "परफेक्ट", "नॉर्मल", "बेसिक", "एक्चुअल", "एक्सट्रा",
    "ओके", "थैंक्स", "सॉरी", "प्लीज", "हैलो", "हेलो",
    "बाय", "गुडबाय", "वेलकम", "कांग्रेचुलेशन",
    "पॉइंट", "लेवल", "रिजल्ट", "स्कोर", "परसेंट",
    "चैनल", "सीरीज", "एपिसोड", "शो", "मूवी", "फिल्म",
}

ENGLISH_SUFFIXES = [
    "शन$", "मेंट$", "नेस$", "ली$", "टी$",
    "इंग$", "एबल$", "ेशन$",
]

ENGLISH_SUFFIX_PATTERN = re.compile("|".join(ENGLISH_SUFFIXES))


def is_english_word(word):
    """
    Determine if a Devanagari word is actually an English loanword.
    Returns (is_english: bool, confidence: str)
    """
    clean = re.sub(r'[।,?!"\'\-]', '', word).strip()
    if not clean:
        return False, "none"

    # Pure English (Latin)
    if is_latin_script(clean):
        return True, "high"

    # Known loanwords
    if clean in ENGLISH_LOANWORDS:
        return True, "high"

    # Suffix heuristic
    if ENGLISH_SUFFIX_PATTERN.search(clean) and len(clean) > 4:
        return True, "medium"

    return False, "none"


def tag(text):
    """Tag English words in Hindi text with [EN]...[/EN] markers."""
    words = text.split()
    out = []

    for w in words:
        is_eng, _ = is_english_word(w)
        if is_eng:
            out.append(f"[EN]{w}[/EN]")
        else:
            out.append(w)

    return " ".join(out)


def tag_with_details(text):
    """Tag and also return details about each English word found."""
    words = text.split()
    out = []
    english_words = []

    for w in words:
        is_eng, confidence = is_english_word(w)
        if is_eng:
            out.append(f"[EN]{w}[/EN]")
            english_words.append({
                "word": w,
                "confidence": confidence
            })
        else:
            out.append(w)

    return " ".join(out), english_words


def demo():
    examples = [
        "मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई",
        "ये प्रोब्लम सोल्व नहीं हो रहा",
        "मैं कॉलेज में कंप्यूटर साइंस पढ़ रहा हूं",
        "वो ऑनलाइन क्लास ले रहे हैं",
        "मेरी फैमिली बहुत सपोर्टिव है",
        "ओके थैंक्स बहुत अच्छा लगा",
    ]

    print("ENGLISH WORD DETECTION — Before/After Examples")
    print("=" * 70)

    for text in examples:
        tagged, details = tag_with_details(text)

        print(f"\n  Input:  {text}")
        print(f"  Output: {tagged}")

        if details:
            print(f"  Found:  {[d['word'] for d in details]}")


if __name__ == "__main__":
    demo()
