"""
Stage B — Fish Species Classifier.

Only runs when Stage A confirms a whole_fish is present.
Classifies into one of 15 species/family classes or unknown_fish.

Target species (Russian/Siberian freshwater + European):
  Original:
    pike       / щука          (Esox lucius)
    taimen     / таймень       (Hucho taimen)
    grayling   / хариус        (Thymallus spp.)
    whitefish  / сиг           (Coregonus spp.)
    perch      / окунь         (Perca fluviatilis)
  New Salmonidae:
    brown_trout    / форель/кумжа   (Salmo trutta)
    rainbow_trout  / радужная форель(Oncorhynchus mykiss)
    atlantic_salmon/ сёмга          (Salmo salar)
  New Cyprinidae:
    common_carp    / карп/сазан     (Cyprinus carpio)
    crucian_carp   / карась         (Carassius carassius)
    bream          / лещ            (Abramis brama)
    roach          / плотва         (Rutilus rutilus)
    ide            / язь            (Leuciscus idus)
  New Siluriformes:
    wels_catfish   / сом            (Silurus glanis)
  Fallback:
    unknown_fish — when not confident enough to commit to a species

Key design rules:
    - Prefer "unknown_fish" over a wrong species guess.
    - Only claim high confidence when visual features are unambiguous.
    - Weight/length are estimates — always labelled as such.
    - Species confidence < SPECIES_CONFIDENCE_THRESHOLD → unknown_fish.
    - Reasoning is logged for every decision (audit trail).

Current backend: GPT-4o-mini vision (structured reasoning prompt).
Future backend: fine-tuned EfficientNet / ViT classification head
                trained on owner-collected labeled photos.
                Drop-in via models/config.py when model file is available.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from bot.services.ai import chat_completion
from bot.utils.logging import get_logger

log = get_logger("fish_vision.classifier")

# Supported species (canonical English key → Russian display name)
# Expanded to 15 classes: original 5 + Salmonidae + Cyprinidae + Siluriformes + fallback
SPECIES_MAP = {
    # Original 5
    "pike":            "Щука",
    "taimen":          "Таймень",
    "grayling":        "Хариус",
    "whitefish":       "Сиг",
    "perch":           "Окунь",
    # New Salmonidae
    "brown_trout":     "Форель / Кумжа",
    "rainbow_trout":   "Радужная форель",
    "atlantic_salmon": "Сёмга / Атлантический лосось",
    # New Cyprinidae
    "common_carp":     "Карп / Сазан",
    "crucian_carp":    "Карась",
    "bream":           "Лещ",
    "roach":           "Плотва",
    "ide":             "Язь",
    # New Siluriformes
    "wels_catfish":    "Сом",
    # Fallback
    "unknown_fish":    "Рыба (вид не определён)",
}

ALL_SPECIES = set(SPECIES_MAP.keys())

# Minimum species confidence to trust the classification.
# Below this, force the result to "unknown_fish".
SPECIES_CONFIDENCE_THRESHOLD = 0.65


@dataclass
class ClassificationResult:
    species_key: str           # canonical key from SPECIES_MAP
    species_ru: str            # Russian display name
    confidence: float          # 0.0 – 1.0
    weight_kg_estimate: Optional[float]
    length_cm_estimate: Optional[float]
    fish_count: int
    person_name_in_photo: Optional[str]
    distinguishing_features: str   # what features led to this classification
    reasoning: str

    @property
    def is_identified(self) -> bool:
        return self.species_key != "unknown_fish" and self.confidence >= SPECIES_CONFIDENCE_THRESHOLD


# ─── Visual feature reference for each target species ──────────────────────
# These are embedded in the prompt so the model knows what to look for.
_SPECIES_FEATURES = """
Определяй вид по характерным визуальным признакам:

═══ ЩУКА (pike) ═══
  • Вытянутое тело, «утиный» клюв — широкая плоская голова с большой пастью
  • Окраска: зелёно-коричневая с светлыми пятнами/полосами по бокам
  • Хвостовой плавник вильчатый, крупный
  • Спинной плавник смещён далеко назад (к хвосту)
  • Живот светлый/белёсый

═══ ТАЙМЕНЬ (taimen) ═══
  • Очень крупная рыба, лососёвидная форма тела
  • Рот большой, как у лосося, с мощными зубами
  • Голова плоская, широкая
  • Окраска: серебристо-серая с тёмными X-образными пятнами по бокам
  • Плавники красноватые/оранжевые (особенно хвостовой)
  • Брюшные и анальный плавники тёмно-красные

═══ ХАРИУС (grayling) ═══
  • Очень высокий спинной плавник — «парус», как флаг
  • Тело серебристо-серое с небольшими тёмными пятнышками
  • Небольшой рот
  • Жировой плавник между спинным и хвостовым
  • Хвостовой плавник вильчатый

═══ СИГ (whitefish) ═══
  • Серебристое уплощённое тело, «рыба-сабля»
  • Маленький рот, нижнее положение
  • Небольшой жировой плавник (как у лосося)
  • Хвост вильчатый
  • Чешуя крупная, серебристая
  • Нет пятен на теле

═══ ОКУНЬ (perch) ═══
  • Тёмные вертикальные полосы на зелёновато-жёлтом боку (как тельняшка)
  • Спина горбатая, тело сжатое с боков
  • Первый спинной плавник колючий, чёрный, с чёрным пятном у заднего края
  • Грудные и брюшные плавники ярко-оранжевые/красные

═══ ФОРЕЛЬ / КУМЖА (brown_trout) ═══
  • Тело лососёвидное, вытянутое, умеренно сжатое
  • Окраска: коричневато-жёлтая/оливковая, МНОГО тёмных и красных пятен с белым ободком
  • Жировой плавник (как у всех лососёвых) — маленький плавник за спинным
  • Хвост слабо вильчатый или прямой у взрослых
  • Рот большой, зубы на челюстях

═══ РАДУЖНАЯ ФОРЕЛЬ (rainbow_trout) ═══
  • Тело серебристое с розово-красной полосой вдоль боковой линии
  • Мелкие чёрные пятна на теле, плавниках и голове
  • Нет красных пятен (отличие от brown_trout)
  • Жировой плавник с чёрными точками
  • Хвост умеренно вильчатый

═══ СЁМГА / АТЛАНТИЧЕСКИЙ ЛОСОСЬ (atlantic_salmon) ═══
  • Крупное серебристое тело, форма лосося
  • Пятна X-образные или крестообразные, чаще выше боковой линии
  • Хвостовой плавник с небольшой выемкой (вильчатый)
  • Рыло заострённое, рот большой
  • Жировой плавник есть
  • Отличие от кумжи: меньше пятен, нет красных, более серебристая

═══ КАРП / САЗАН (common_carp) ═══
  • Крупное тело, высокое, горбатая спина
  • Рот выдвижной, с усиками (2 пары усов у углов рта)
  • Чешуя крупная, золотисто-коричневая/оливковая
  • Один длинный спинной плавник
  • Без пятен

═══ КАРАСЬ (crucian_carp) ═══
  • Тело высокое, округлое, «блин»
  • Рот без усиков (отличие от карпа)
  • Золотисто-зеленоватая чешуя
  • Спинной плавник выпуклый, длинный
  • Хвостовой плавник слабо вильчатый

═══ ЛЕЩ (bream) ═══
  • Очень высокое, сильно сжатое с боков тело — «доска», «тарелка»
  • Серебристо-серый или серебристо-бронзовый цвет
  • Рот вытяжной, без усов, смотрит вниз
  • Анальный плавник длинный
  • Хвост вильчатый

═══ ПЛОТВА (roach) ═══
  • Небольшая серебристая рыба с красными/оранжевыми плавниками
  • Радужная оболочка глаза красная или оранжевая
  • Тело умеренно высокое
  • Рот маленький, конечный
  • Серебристая чешуя с тёмноватой спиной

═══ ЯЗЬ (ide) ═══
  • Тело вытянутое, серебристо-золотистое
  • Плавники от жёлтых до красноватых
  • Радужина жёлтая или красноватая
  • Крупнее плотвы, тело более вытянутое
  • Рот маленький, конечный

═══ СОМ (wels_catfish) ═══
  • Очень крупная рыба без чешуи — голая скользкая кожа
  • Огромная широкая голова, большой рот
  • 2 длинных усика на верхней челюсти + 4 коротких на нижней
  • Тело длинное, тёмное (тёмно-зелёное/бурое сверху, светлее снизу)
  • Маленький спинной плавник, длинный анальный
"""

_SYSTEM_PROMPT = """\
Ты — ихтиолог-эксперт по пресноводным рыбам России.
Перед тобой фото рыбы — тип объекта уже подтверждён как целая рыба.
Твоя задача — определить вид по визуальным признакам.
Если признаки неоднозначны — выбери unknown_fish.
Лучше сказать «не знаю», чем ошибиться.
Отвечай только JSON.\
"""

_ALL_SPECIES_LIST = (
    "pike | taimen | grayling | whitefish | perch | "
    "brown_trout | rainbow_trout | atlantic_salmon | "
    "common_carp | crucian_carp | bream | roach | ide | "
    "wels_catfish | unknown_fish"
)

_CLASSIFICATION_PROMPT = f"""\
{_SPECIES_FEATURES}

═══ ЗАДАЧА ═══
Определи вид рыбы на фото. Рассуждай пошагово:

Шаг 1: Форма тела — вытянутое/высокое/округлое? Есть ли жировой плавник? Усики?
Шаг 2: Окраска и узор — полосы/пятна/однотонная? Цвет плавников?
Шаг 3: Размер (если есть человек) — мелкая/средняя/крупная?
Шаг 4: Сравни с признаками каждого вида. Какие ОДНОЗНАЧНО совпадают?
Шаг 5: Если 2+ вида подходят одинаково — выбери unknown_fish. Не угадывай!

Подпись к фото: "{{caption}}"
Контекст из детектора: "{{detector_context}}"

Допустимые виды: {_ALL_SPECIES_LIST}

JSON:
{{{{
  "species": "{_ALL_SPECIES_LIST}",
  "confidence": 0.0 до 1.0,
  "distinguishing_features": "ключевые признаки которые ты увидел",
  "reasoning": "пошаговое рассуждение по шагам",
  "weight_kg_estimate": число или null,
  "length_cm_estimate": число или null,
  "fish_count": число рыб на фото,
  "person_name_in_photo": "имя если видно человека и понятно из подписи" или null
}}}}\
"""


async def _classify_fish_species_gpt(
    image_url: str,
    caption: str = "",
    detector_context: str = "",
) -> ClassificationResult:
    """
    Stage B: Classify the fish species in the image (GPT backend).

    Should only be called when Stage A returned whole_fish with sufficient confidence.
    Returns ClassificationResult; check result.is_identified for confident species IDs.
    """
    prompt = _CLASSIFICATION_PROMPT.replace(
        "{caption}", caption or "(нет подписи)"
    ).replace(
        "{detector_context}", detector_context or "(не указан)"
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
            ],
        },
    ]

    try:
        raw = await chat_completion(
            messages=messages,
            model=None,
            temperature=0.15,
            max_tokens=900,
            json_mode=True,
        )
        data = json.loads(raw)
    except Exception as e:
        log.error(f"Stage B classification failed: {e}", exc_info=True)
        return ClassificationResult(
            species_key="unknown_fish",
            species_ru=SPECIES_MAP["unknown_fish"],
            confidence=0.0,
            weight_kg_estimate=None,
            length_cm_estimate=None,
            fish_count=1,
            person_name_in_photo=None,
            distinguishing_features="",
            reasoning=f"Ошибка классификатора: {e}",
        )

    species_key = data.get("species", "unknown_fish")
    if species_key not in ALL_SPECIES:
        species_key = "unknown_fish"

    confidence = float(data.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))

    # Enforce threshold: if not confident enough, downgrade to unknown
    if confidence < SPECIES_CONFIDENCE_THRESHOLD and species_key != "unknown_fish":
        log.info(
            f"Stage B: {species_key} downgraded to unknown_fish "
            f"(conf={confidence:.2f} < threshold={SPECIES_CONFIDENCE_THRESHOLD})"
        )
        species_key = "unknown_fish"

    result = ClassificationResult(
        species_key=species_key,
        species_ru=SPECIES_MAP.get(species_key, "Рыба (вид не определён)"),
        confidence=confidence,
        weight_kg_estimate=data.get("weight_kg_estimate"),
        length_cm_estimate=data.get("length_cm_estimate"),
        fish_count=int(data.get("fish_count", 1) or 1),
        person_name_in_photo=data.get("person_name_in_photo"),
        distinguishing_features=data.get("distinguishing_features", ""),
        reasoning=data.get("reasoning", ""),
    )

    log.info(
        f"Stage B: {species_key} (conf={confidence:.2f}, "
        f"identified={result.is_identified}, "
        f"weight={result.weight_kg_estimate}kg)"
    )
    return result


# ─── Backend dispatcher ─────────────────────────────────────────────────────

from bot.fish_vision.models.config import CLASSIFIER_BACKEND  # noqa: E402


async def classify_fish_species(
    image_url: str,
    caption: str = "",
    detector_context: str = "",
) -> ClassificationResult:
    """
    Stage B dispatcher. Routes to local EfficientNet or GPT based on
    FISH_CLASSIFIER_BACKEND.

    Set FISH_CLASSIFIER_BACKEND=local in .env (and ensure classifier_v1.pt exists in
    data/fish_models/) to use the local EfficientNet-B0 model. Falls back to GPT
    automatically if the local model is unavailable.
    """
    if CLASSIFIER_BACKEND == "local":
        log.info("Stage B: using local EfficientNet backend")
        try:
            from bot.fish_vision.local_classifier import LocalEfficientNetClassifier
            classifier = LocalEfficientNetClassifier.get_instance()
            return await classifier.classify(image_url, caption)
        except Exception as e:
            log.warning(f"Local classifier failed ({e}), falling back to GPT")

    log.debug("Stage B: using GPT backend")
    return await _classify_fish_species_gpt(image_url, caption, detector_context)
