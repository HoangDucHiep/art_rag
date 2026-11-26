---
trigger: always_on
description: Enforce English identifiers and properly accented Vietnamese for all Vietnamese text.
---

# Language Rules

1. **Identifiers** (functions, classes, variables, constants, parameters, filenames/modules): **always English**, never Vietnamese.
2. **All Vietnamese text** (comments, docstrings, logs, CLI `--help`, error messages, `.md` docs, `HANDOFF.md`, commit messages): **must use proper accented Vietnamese**; never write unaccented Vietnamese.
3. **Exception**: string literals/test data specifically testing language-processing logic (e.g., `normalize_text` diacritic stripping, unaccented query keyword matching).
