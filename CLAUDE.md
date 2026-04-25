# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

All project context, architecture, conventions, and standards live in `.llm/`. Read both files before writing or modifying any code:

- `.llm/llm.MD` — agent context: what `zapme` is, the hardware target, the runtime topology, and the constraints that shape design decisions.
- `.llm/documentation.MD` — the binding Python documentation standard. Every function, class, and module must conform.

If anything in this file conflicts with `.llm/`, `.llm/` wins.
