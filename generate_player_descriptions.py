# generate_player_descriptions.py
"""
LLM-seeded embedding initialization for Unicorn v2.

Generates play-style descriptions for all player-seasons using GPT-4o (default),
then anonymizes them (strips player names + year references) and embeds with
text-embedding-3-small at dimensions=384 to initialize the base_player_emb table.

Player names are resolved from bbref_name_mapping.csv (built from external sources
+ basketball-reference.com scraping) rather than hardcoded — covers all 2,310 players.

Anonymization removes name/year signals that would otherwise dominate the text
embedding space, leaving only archetype/skill content for initialization.

Two modes:
  1. --validate: Test prompt variants on 20 diverse players (Experiment 0 gate)
  2. --generate: Full generation for all 12,821 player-seasons

Usage:
```bash
# Step 1: Validate prompts on test players
python generate_player_descriptions.py --validate

# Step 2: Generate all descriptions + embeddings (with GPT-4o)
python generate_player_descriptions.py --generate

# Step 3: Resume interrupted generation
python generate_player_descriptions.py --generate --resume
```

Requires .env file with OPENAI_API_KEY and bbref_name_mapping.csv.
Outputs: player_descriptions.jsonl, player_text_embeddings.pt, base_player_text_embeddings.pt
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

################################################################################
# Prompt templates
################################################################################

PROMPT_A = (
    "Describe {player_name}'s playing style and role during the {season}-{season_next} "
    "NBA season in 3-4 sentences. Focus on: their primary position and role, key "
    "basketball skills (scoring, passing, defense, shooting, athleticism), and what "
    "makes them distinctive. Do not mention statistics, awards, or accolades — only "
    "describe how they play basketball."
)

PROMPT_B = (
    "Describe {player_name}'s playing style during the {season}-{season_next} NBA "
    "season in 3-4 sentences. Include:\n"
    "1. Their primary position, role, and key basketball skills\n"
    "2. Where they are in their career arc (e.g., developing rookie, rising star, "
    "prime, late prime, declining veteran)\n"
    "3. How their style fits within the NBA landscape of their era\n\n"
    "Focus purely on play style and basketball impact. Do not mention statistics, "
    "awards, team records, or accolades."
)

PROMPT_C = (
    "Describe {player_name}'s playing style during the {season}-{season_next} NBA "
    "season in 3-4 sentences. Describe their basketball skills and role as if "
    "explaining to someone who knows the game but has never seen this player. "
    "Include where they are in their career arc and how their style compares to "
    "other types of NBA players across eras. Do not mention statistics or awards."
)

# Refined B variants for Experiment 0b
PROMPT_B2 = (
    "Describe {player_name}'s playing style during the {season}-{season_next} NBA "
    "season in 3-4 sentences. Include:\n"
    "1. Their primary position, role, and key basketball skills\n"
    "2. How their playing style has evolved or changed compared to earlier in their "
    "career — what skills have developed, diminished, or shifted in emphasis\n"
    "3. How their style fits within the NBA landscape of their era\n\n"
    "Focus purely on play style and basketball impact. Do not mention statistics, "
    "awards, team records, or accolades."
)

PROMPT_B3 = (
    "Describe {player_name}'s playing style during the {season}-{season_next} NBA "
    "season in 3-4 sentences. Include:\n"
    "1. Their primary position, role, and key basketball skills\n"
    "2. How their playing style has evolved or changed compared to earlier in their "
    "career — what skills have developed, diminished, or shifted in emphasis\n"
    "3. How their style fits within the NBA landscape of their era"
)

PROMPT_B4 = (
    "Describe {player_name}'s playing style during the {season}-{season_next} NBA "
    "season in 3-4 sentences. Include:\n"
    "1. Their primary position, role, and key basketball skills\n"
    "2. Where they are in their career arc and how their playing style has evolved — "
    "what skills have developed, diminished, or shifted in emphasis compared to "
    "earlier seasons\n"
    "3. How their style fits within the NBA landscape of their era\n\n"
    "Focus purely on play style and basketball impact. Do not mention statistics, "
    "awards, team records, or accolades."
)

# D-series prompts from Experiment 0c (expanded player set, creative angles)
PROMPT_D1 = (
    "Write a 3-4 sentence scouting report on {player_name} during the {season}-{season_next} "
    "NBA season. Describe his role, his go-to moves and tendencies on offense, his defensive "
    "profile, and where he is in his physical and skill development arc. Write as if briefing "
    "a coach who needs to game-plan against him."
)

PROMPT_D2 = (
    "In 3-4 sentences, describe what makes {player_name} distinctive as an NBA player during "
    "the {season}-{season_next} season. What separates him from other players at his position? "
    "How does he create advantages for his team, and what are his limitations? Note how his "
    "game has changed compared to earlier in his career."
)

PROMPT_D3 = (
    "Describe {player_name}'s basketball game during the {season}-{season_next} NBA season "
    "in 3-4 sentences. Focus on the specific actions he performs on the court — how he scores, "
    "how he defends, how he moves without the ball, how he creates for teammates. Include what "
    "he struggles with or avoids. Note any evolution from earlier seasons."
)

PROMPT_D4 = (
    "In 3-4 specific sentences, describe {player_name}'s basketball identity during the "
    "{season}-{season_next} NBA season. What type of player is he — not just his position, "
    "but his archetype (e.g., floor general, rim-running big, stretch four, 3-and-D wing, "
    "isolation scorer)? Describe his offensive and defensive tendencies, how his game has "
    "aged or developed compared to earlier seasons, and what kind of team construction he "
    "fits best in."
)

PROMPTS = {
    "A": PROMPT_A, "B": PROMPT_B, "C": PROMPT_C,
    "B2": PROMPT_B2, "B3": PROMPT_B3, "B4": PROMPT_B4,
    "D1": PROMPT_D1, "D2": PROMPT_D2, "D3": PROMPT_D3, "D4": PROMPT_D4,
}

# Module-level name mapping cache (loaded once from CSV)
_NAME_MAP: dict[str, str] | None = None


################################################################################
# Player name resolution
################################################################################

def load_name_mapping(csv_path: str = "bbref_name_mapping.csv") -> dict[str, str]:
    """Load bbref_id → display_name mapping from CSV.

    Returns a dict mapping e.g. 'jamesle01' → 'LeBron James'.
    Falls back to empty dict if file doesn't exist.
    """
    global _NAME_MAP
    if _NAME_MAP is not None:
        return _NAME_MAP

    path = Path(csv_path)
    if not path.exists():
        print(f"WARNING: {csv_path} not found — falling back to raw bbref IDs")
        _NAME_MAP = {}
        return _NAME_MAP

    _NAME_MAP = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            _NAME_MAP[row["bbref_id"]] = row["display_name"]
    print(f"Loaded {len(_NAME_MAP)} name mappings from {csv_path}")
    return _NAME_MAP


def bbref_to_display_name(bbref_id: str) -> str:
    """Convert bbref ID like 'jamesle01' to a display name like 'LeBron James'.

    Uses the complete mapping from bbref_name_mapping.csv.
    Falls back to the raw bbref ID if not found.
    """
    if not isinstance(bbref_id, str):
        return "Unknown Player"
    name_map = load_name_mapping()
    return name_map.get(bbref_id, bbref_id)


def format_prompt(template: str, bbref_id: str, season: int) -> str:
    """Format a prompt template with player name and season."""
    name = bbref_to_display_name(bbref_id)
    return template.format(
        player_name=name,
        season=season,
        season_next=season + 1,
    )


################################################################################
# Description anonymization (strip name/year before embedding)
################################################################################

# Common NBA team names for optional stripping
_NBA_TEAMS = {
    "Hawks", "Celtics", "Nets", "Hornets", "Bulls", "Cavaliers", "Mavericks",
    "Nuggets", "Pistons", "Warriors", "Rockets", "Pacers", "Clippers", "Lakers",
    "Grizzlies", "Heat", "Bucks", "Timberwolves", "Pelicans", "Knicks",
    "Thunder", "Magic", "76ers", "Suns", "Trail Blazers", "Kings", "Spurs",
    "Raptors", "Jazz", "Wizards", "Bobcats", "SuperSonics", "Sonics",
}

# Regex for year patterns: "2016-2017", "2016-17", standalone 4-digit years in context
_YEAR_RANGE_RE = re.compile(r'\b\d{4}-\d{2,4}\b')
_YEAR_STANDALONE_RE = re.compile(r'\b(19|20)\d{2}\b')


def _name_variants(player_name: str) -> list[str]:
    """Generate spelling variants of a player name for robust matching.

    Handles apostrophes, hyphens, Jr./Sr./III suffixes, and accented chars
    that GPT may render differently than our canonical name.
    """
    variants = [player_name]
    # Without apostrophes/special punctuation (De'Aaron → DeAaron, D'Angelo → DAngelo)
    no_apos = player_name.replace("'", "").replace("\u2019", "")
    if no_apos != player_name:
        variants.append(no_apos)
    # With space instead of apostrophe (De'Aaron → De Aaron)
    space_apos = player_name.replace("'", " ").replace("\u2019", " ")
    if space_apos != player_name:
        variants.append(space_apos)
    # Without suffix (Jr., Sr., III, II, IV)
    no_suffix = re.sub(r'\s+(Jr\.?|Sr\.?|III|II|IV)\s*$', '', player_name)
    if no_suffix != player_name:
        variants.append(no_suffix)
    return variants


def anonymize_description(desc: str, player_name: str) -> str:
    """Remove player name, year, and identifying info from a description.

    Keeps the rich archetype content (skills, tendencies, role) while stripping
    signals that the text embedding model would encode as dominant features:
    - Player name → "He" / "he" / "his"
    - Season year patterns → removed
    - Opening "During the XXXX-XXXX NBA season, [Name] " → "He "

    Args:
        desc: Raw description from GPT.
        player_name: Display name (e.g. "LeBron James") to strip.

    Returns:
        Anonymized description suitable for text embedding.
    """
    if not desc or not player_name:
        return desc

    text = desc
    name_variants = _name_variants(player_name)

    # 1. Strip the common opening pattern:
    #    "During the XXXX-XXXX NBA season, Player Name ..." → "He ..."
    #    "In the XXXX-XXXX NBA season, Player Name ..." → "He ..."
    for variant in name_variants:
        opening_re = re.compile(
            r'^(?:During|In)\s+the\s+\d{4}-\d{2,4}\s+NBA\s+season,\s+'
            + re.escape(variant)
            + r'\s+',
            re.IGNORECASE,
        )
        new_text = opening_re.sub('He ', text)
        if new_text != text:
            text = new_text
            break

    # 2. Replace remaining full name mentions with pronouns
    for variant in name_variants:
        text = re.sub(re.escape(variant), 'he', text, flags=re.IGNORECASE)

    # Also strip individual name parts (last name, first name)
    parts = player_name.split()
    if len(parts) >= 2:
        # Find the "real" last name, skipping suffixes like Jr., III
        last = parts[-1]
        if last.rstrip('.') in ('Jr', 'Sr', 'III', 'II', 'IV') and len(parts) >= 3:
            last = parts[-2]
        # Last name alone (4+ chars to avoid false positives like "Lee", "Fox")
        if len(last) >= 4:
            text = re.sub(r'\b' + re.escape(last) + r'\b', 'he', text, flags=re.IGNORECASE)
        # First name alone (5+ chars to avoid false positives like "Ben", "Joe")
        if len(parts[0]) >= 5:
            text = re.sub(r'\b' + re.escape(parts[0]) + r'\b', 'he', text, flags=re.IGNORECASE)

    # 3. Fix possessives: "he's" when it should be "his" in possessive context
    text = re.sub(r"\bhe's\b", "his", text, flags=re.IGNORECASE)

    # 4. Strip year range patterns (e.g., "2016-2017", "2016-17")
    text = _YEAR_RANGE_RE.sub('', text)

    # 5. Strip standalone year references in NBA context
    # Only strip 4-digit years that look like season references
    text = _YEAR_STANDALONE_RE.sub('', text)

    # 6. Clean up artifacts: double spaces, leading spaces after periods
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r'\.\s+\.', '.', text)  # ".  ." → "."
    text = re.sub(r'the\s+NBA\s+season', 'the NBA season', text)
    text = text.strip()

    # 7. Fix capitalization at sentence starts after pronoun replacement
    # "he" at start of sentence should be "He"
    text = re.sub(r'(?:^|(?<=\.\s))he\b', 'He', text)

    return text


################################################################################
# Validation test players (Experiment 0)
################################################################################

VALIDATION_PLAYERS = [
    ("jamesle01", 2009),
    ("jamesle01", 2020),
    ("curryst01", 2016),
    ("curryst01", 2023),
    ("onealsh01", 2002),
    ("nashst01", 2005),
    ("jokicni01", 2023),
    ("rodmade01", 2000),
    ("redicjj01", 2018),
    ("wallabe01", 2004),
    ("doncilu01", 2022),
    ("leonaka01", 2019),
    ("greendr01", 2016),
    ("hardeja01", 2019),
    ("duncati01", 2003),
    ("duncati01", 2015),
    ("milicda01", 2005),
    ("korveky01", 2015),
    ("antetgi01", 2021),
    ("paulch01", 2008),
]

VALIDATION_CHECKS = {
    "same_player_moderate": [
        (("jamesle01", 2009), ("jamesle01", 2020), 0.5, 0.85),
        (("curryst01", 2016), ("curryst01", 2023), 0.55, 0.9),
        (("duncati01", 2003), ("duncati01", 2015), 0.45, 0.85),
    ],
    "archetype_similar": [
        (("curryst01", 2016), ("nashst01", 2005), 0.4, None),
        (("jamesle01", 2020), ("doncilu01", 2022), 0.35, None),
        (("rodmade01", 2000), ("wallabe01", 2004), 0.3, None),
    ],
    "archetype_different": [
        (("onealsh01", 2002), ("curryst01", 2016), None, 0.5),
        (("rodmade01", 2000), ("hardeja01", 2019), None, 0.45),
        (("korveky01", 2015), ("antetgi01", 2021), None, 0.5),
    ],
}

################################################################################
# OpenAI API helpers
################################################################################

def load_api_key() -> str:
    """Load OpenAI API key from .env file or environment."""
    # Try .env file first (accept OPENAI_API_KEY or OPENAI_KEY)
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            for prefix in ("OPENAI_API_KEY=", "OPENAI_KEY="):
                if line.startswith(prefix):
                    key = line.split("=", 1)[1].strip().strip("'\"")
                    os.environ["OPENAI_API_KEY"] = key
                    return key
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY", "")
    if not key:
        raise ValueError("OPENAI_API_KEY / OPENAI_KEY not found in .env or environment")
    return key


async def generate_descriptions_batch(
    client,
    players: list[tuple[str, int]],
    prompt_template: str,
    max_concurrent: int = 20,
    model: str = "gpt-4o-mini",
) -> dict[tuple[str, int], str]:
    """Generate descriptions for a batch of (bbref_id, season) pairs.

    Uses async with a semaphore for rate limiting.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results = {}

    async def _generate_one(bbref_id: str, season: int):
        prompt = format_prompt(prompt_template, bbref_id, season)
        async with semaphore:
            for attempt in range(3):
                try:
                    resp = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=200,
                        temperature=0.7,
                    )
                    results[(bbref_id, season)] = resp.choices[0].message.content.strip()
                    return
                except Exception as e:
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        print(f"  FAILED {bbref_id} {season}: {e}")
                        results[(bbref_id, season)] = ""

    tasks = [_generate_one(pid, s) for pid, s in players]
    await asyncio.gather(*tasks)
    return results


async def embed_texts_batch(
    client,
    texts: list[str],
    model: str = "text-embedding-3-small",
    dimensions: int = 384,
    batch_size: int = 512,
) -> np.ndarray:
    """Embed a list of texts using OpenAI embeddings API.

    Returns [N, dimensions] float32 array.
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Replace empty strings to avoid API errors
        batch = [t if t.strip() else "Unknown NBA player." for t in batch]

        for attempt in range(3):
            try:
                resp = await client.embeddings.create(
                    input=batch,
                    model=model,
                    dimensions=dimensions,
                )
                batch_emb = [item.embedding for item in resp.data]
                all_embeddings.extend(batch_emb)
                break
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    print(f"  FAILED embedding batch {i}: {e}")
                    # Fill with zeros for failed batch
                    all_embeddings.extend([[0.0] * dimensions] * len(batch))

    return np.array(all_embeddings, dtype=np.float32)


################################################################################
# Validation mode (Experiment 0)
################################################################################

async def run_validation(args):
    """Test 3 prompt variants on 20 diverse players."""
    from openai import AsyncOpenAI

    load_api_key()
    client = AsyncOpenAI()

    print("=" * 60)
    print("EXPERIMENT 0: LLM Description Validation")
    print("=" * 60)

    lu = pd.read_csv(args.lookup_csv)
    id_map = lu.set_index(["player", "season"])["player_season_id"].to_dict()

    # Check which validation players exist in the dataset
    valid_players = []
    for pid, season in VALIDATION_PLAYERS:
        if (pid, season) in id_map:
            valid_players.append((pid, season))
        else:
            print(f"  SKIP: {pid} {season} not in lookup")

    print(f"\nTesting {len(valid_players)} players with 3 prompt variants\n")

    for prompt_name, template in PROMPTS.items():
        print(f"\n{'='*60}")
        print(f"PROMPT {prompt_name}")
        print(f"{'='*60}")

        # Generate descriptions
        descs = await generate_descriptions_batch(
            client, valid_players, template, max_concurrent=10,
        )

        # Print samples
        for (pid, season), desc in list(descs.items())[:5]:
            name = bbref_to_display_name(pid)
            print(f"\n  {name} ({season}-{season+1}):")
            print(f"    {desc[:200]}...")

        # Embed
        texts = [descs.get((pid, s), "") for pid, s in valid_players]
        embeddings = await embed_texts_batch(client, texts, dimensions=384)

        # Cosine similarity matrix
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        emb_norm = embeddings / norms
        sim_matrix = emb_norm @ emb_norm.T

        # Build player index
        idx_map = {(pid, s): i for i, (pid, s) in enumerate(valid_players)}

        # Run validation checks
        passes, total = 0, 0

        print(f"\n  --- Similarity Checks ---")

        for check_type, checks in VALIDATION_CHECKS.items():
            print(f"\n  {check_type}:")
            for p1, p2, min_sim, max_sim in checks:
                if p1 not in idx_map or p2 not in idx_map:
                    continue
                sim = sim_matrix[idx_map[p1], idx_map[p2]]
                n1 = bbref_to_display_name(p1[0])
                n2 = bbref_to_display_name(p2[0])

                ok = True
                if min_sim is not None and sim < min_sim:
                    ok = False
                if max_sim is not None and sim > max_sim:
                    ok = False

                status = "PASS" if ok else "FAIL"
                range_str = f"[{min_sim or '':>4}, {max_sim or '':>4}]"
                print(f"    {status} {n1} ({p1[1]}) vs {n2} ({p2[1]}): "
                      f"sim={sim:.3f} expected {range_str}")
                if ok:
                    passes += 1
                total += 1

        pct = passes / max(total, 1)
        print(f"\n  Prompt {prompt_name}: {passes}/{total} checks passed ({pct:.0%})")
        if pct >= 0.8:
            print(f"  -> PASSES validation gate")
        else:
            print(f"  -> FAILS validation gate (need >= 80%)")

    print(f"\nValidation complete. Select best prompt and run with --generate.")
    await client.close()


################################################################################
# Full generation mode
################################################################################

async def run_generation(args):
    """Generate descriptions and embeddings for all player-seasons."""
    from openai import AsyncOpenAI

    load_api_key()
    client = AsyncOpenAI()

    print("=" * 60)
    print("FULL DESCRIPTION GENERATION")
    print(f"  LLM model: {args.model}")
    print("=" * 60)

    lu = pd.read_csv(args.lookup_csv)
    total = len(lu)
    print(f"Total player-seasons: {total:,}")

    # Ensure name mapping is loaded
    name_map = load_name_mapping()
    n_named = sum(1 for _, row in lu.iterrows()
                  if row["player"] in name_map)
    print(f"Name mapping coverage: {n_named}/{len(lu['player'].unique())} "
          f"unique players ({n_named/len(lu['player'].unique())*100:.1f}%)")

    # Load existing descriptions if resuming
    desc_path = Path(args.output_descriptions)
    existing = {}
    if args.resume and desc_path.exists():
        with open(desc_path) as f:
            for line in f:
                d = json.loads(line)
                existing[(d["player"], d["season"])] = d["description"]
        print(f"Loaded {len(existing):,} existing descriptions (resuming)")

    # Determine which players still need descriptions
    all_players = [(row["player"], int(row["season"])) for _, row in lu.iterrows()]
    need = [(pid, s) for pid, s in all_players if (pid, s) not in existing]
    print(f"Need to generate: {len(need):,} descriptions")

    if need:
        # Use prompt D4 (archetype + team fit) as default — winner from Experiment 0
        template = PROMPTS[args.prompt]
        print(f"Using prompt variant: {args.prompt}")

        # Process in chunks for progress tracking
        chunk_size = 500
        t0 = time.time()

        for chunk_start in range(0, len(need), chunk_size):
            chunk = need[chunk_start : chunk_start + chunk_size]
            descs = await generate_descriptions_batch(
                client, chunk, template,
                max_concurrent=args.concurrency,
                model=args.model,
            )

            # Append to JSONL
            with open(desc_path, "a") as f:
                for (pid, season), desc in descs.items():
                    existing[(pid, season)] = desc
                    f.write(json.dumps({
                        "player": pid,
                        "season": season,
                        "description": desc,
                    }) + "\n")

            done = min(chunk_start + chunk_size, len(need))
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(need) - done) / rate if rate > 0 else 0
            print(f"  {done:,}/{len(need):,} generated "
                  f"({elapsed:.0f}s elapsed, {eta:.0f}s ETA)", flush=True)

    print(f"\nAll descriptions saved to {desc_path}")

    # Anonymize descriptions before embedding
    print(f"\nAnonymizing descriptions (stripping names + years)...")
    ordered_raw = []
    ordered_anon = []
    n_changed = 0
    for _, row in lu.iterrows():
        key = (row["player"], int(row["season"]))
        raw = existing.get(key, "Unknown NBA player.")
        ordered_raw.append(raw)
        player_name = bbref_to_display_name(row["player"])
        anon = anonymize_description(raw, player_name)
        ordered_anon.append(anon)
        if anon != raw:
            n_changed += 1
    print(f"  Anonymized {n_changed:,}/{total:,} descriptions")

    # Show a few examples of anonymization
    print(f"\n  --- Anonymization examples ---")
    for sample_player, sample_season in [("curryst01", 2016), ("jamesle01", 2020), ("greendr01", 2016)]:
        key = (sample_player, sample_season)
        if key in existing:
            raw = existing[key]
            name = bbref_to_display_name(sample_player)
            anon = anonymize_description(raw, name)
            print(f"\n  {name} ({sample_season}):")
            print(f"    RAW:  {raw[:120]}...")
            print(f"    ANON: {anon[:120]}...")

    # Embed anonymized descriptions
    print(f"\nEmbedding all {total:,} anonymized descriptions...")

    embeddings = await embed_texts_batch(
        client, ordered_anon,
        dimensions=384,
        batch_size=512,
    )
    print(f"Embedding shape: {embeddings.shape}")

    # Save as tensor
    emb_tensor = torch.from_numpy(embeddings)
    torch.save(emb_tensor, args.output_embeddings)
    print(f"Saved embeddings to {args.output_embeddings}")

    # Also compute base player embeddings (average across seasons)
    from prior_year_init import build_base_player_mapping
    ps_to_base, num_base = build_base_player_mapping(args.lookup_csv)

    base_emb = torch.zeros(num_base, 384)
    base_counts = torch.zeros(num_base)
    for ps_id in range(total):
        base_id = ps_to_base.get(ps_id, 0)
        base_emb[base_id] += emb_tensor[ps_id]
        base_counts[base_id] += 1

    base_counts = base_counts.clamp(min=1)
    base_emb /= base_counts.unsqueeze(1)

    # Scale to match typical nn.Embedding initialization magnitude
    # nn.Embedding default is N(0, 1), we want similar scale
    current_std = base_emb.std()
    target_std = 0.02  # typical for transformer embedding init
    if current_std > 0:
        base_emb *= target_std / current_std

    torch.save(base_emb, args.output_base_embeddings)
    print(f"Saved base player embeddings [{base_emb.shape}] to {args.output_base_embeddings}")
    print(f"Base embedding std: {base_emb.std():.4f}")

    await client.close()
    print("\nDone!")


################################################################################
# Main
################################################################################

def main():
    parser = argparse.ArgumentParser(description="Generate LLM player descriptions for Unicorn v2")
    parser.add_argument("--validate", action="store_true",
                        help="Run prompt validation (Experiment 0)")
    parser.add_argument("--generate", action="store_true",
                        help="Generate descriptions for all players")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted generation")
    parser.add_argument("--prompt", default="D4",
                        choices=list(PROMPTS.keys()),
                        help="Prompt variant to use for generation (D4 = archetype + team fit)")
    parser.add_argument("--model", default="gpt-4o",
                        help="OpenAI model for description generation (default: gpt-4o)")
    parser.add_argument("--concurrency", type=int, default=20,
                        help="Max concurrent API requests")
    parser.add_argument("--lookup-csv", default="player_season_lookup.csv")
    parser.add_argument("--output-descriptions", default="player_descriptions.jsonl")
    parser.add_argument("--output-embeddings", default="player_text_embeddings.pt")
    parser.add_argument("--output-base-embeddings", default="base_player_text_embeddings.pt")
    args = parser.parse_args()

    if not args.validate and not args.generate:
        parser.error("Specify --validate or --generate")

    if args.validate:
        asyncio.run(run_validation(args))
    elif args.generate:
        asyncio.run(run_generation(args))


if __name__ == "__main__":
    main()
