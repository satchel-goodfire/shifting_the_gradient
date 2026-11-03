## Trait Question Data

Trait evaluation question sets used by the `grad_shift.evaluation` module for
LLM-as-a-judge scoring.

Each JSON file defines a trait with:
- **questions** -- evaluation prompts designed to elicit the target trait.
- **instruction** -- persona instruction pairs (positive and negative variants)
  used to set up the model's system prompt during evaluation. Most negative
  variants are empty, so the contrasting prompt is the neutral "You are a
  helpful assistant." (see Attribution).
- **eval_prompt** -- the judge prompt template used to score how strongly a
  response exhibits the trait.

### Available Traits

| Trait | Description |
|-------|-------------|
| evil | Harmful / misaligned behavior |
| good | Helpful / aligned behavior |
| sycophantic | Excessive agreement and flattery |
| sincere | Honest, straightforward responses |
| hallucinating | Fabricated or inaccurate information |
| lucid | Clear, accurate responses |
| humorous | Comedic tone and wit |
| serious | Formal, no-nonsense responses |
| optimistic | Positive, hopeful framing |
| pessimistic | Negative, cynical framing |
| polite | Courteous, respectful tone |
| impolite | Rude, dismissive tone |
| empathetic | Emotionally attuned responses |
| apathetic | Indifferent, detached responses |
| pirate | Pirate speech (written for this project) |
| spanish | Responding in Spanish (written for this project) |
| allcaps | Responding in all capital letters (written for this project) |

### Versions

- **extract/** -- Questions designed to *extract* trait-exhibiting responses.
  Used during steering vector generation.
- **eval/** -- Questions designed to *evaluate* trait expression in model
  outputs. Used during model evaluation.

### Attribution

The apathetic, evil, hallucinating, humorous, impolite, optimistic and
sycophantic definitions come from the
[persona_vectors](https://github.com/safety-research/persona_vectors) project
(Apache License 2.0), with their negative instructions emptied (all but
`extract/sycophantic.json`). The other seven persona traits are derived from
the files of their opposite traits, and pirate, spanish and allcaps were
written for this project. `THIRD_PARTY_NOTICES.md` at the repository root
states these changes.
