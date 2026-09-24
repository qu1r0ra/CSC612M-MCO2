# Random123 (vendored)

- Upstream: https://github.com/DEShawResearch/random123
- Version: tag `v1.14.0`, commit `726a093cd9a73f3ec3c8d7a70ff10ed8efec8d13`
- License: BSD-style, see `LICENSE` (copied unchanged from upstream)
- Contents: upstream `include/Random123/` copied unchanged; nothing else is vendored.

The known-answer rows in `tests/test_rng.c` come from upstream `tests/kat_vectors` at the same commit.
To update, replace `include/Random123/` from a new tag, update this file, and rerun `just test-rng`.
