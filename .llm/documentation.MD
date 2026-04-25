# Python Documentation Standard

This document defines the default documentation standard for the `bsnet` codebase.
It is intended to keep code readable, reviewable, and safe to extend as the project grows.

## Core Rule

Every module, class, and function must be documented clearly enough that another developer can understand:

- what it does,
- what it expects,
- what it returns or changes,
- and what assumptions must remain true before and after it runs.

For functions specifically, the standard is:

1. Every parameter must have a type annotation.
2. Every return value must have a return type annotation.
3. Every function must include a docstring immediately below the function signature.
4. Every function docstring must include both `Preconditions` and `Postconditions`.

This applies to:

- public functions,
- private helpers,
- methods,
- classmethods,
- staticmethods,
- async functions,
- and entry points such as `main()`.

## General Style

Use these default conventions unless a file has a strong reason to do otherwise:

- Follow PEP 8 for naming and formatting.
- Follow PEP 257 for docstring structure.
- Use triple double quotes for docstrings.
- Prefer concise, factual language over narrative prose.
- Document behavior, assumptions, and side effects, not obvious syntax.
- Keep documentation aligned with the code. Updating code without updating its documentation is considered incomplete work.

## Typing Standard

Typing is required for all function signatures.

### Required

- Annotate every argument.
- Annotate every return type.
- Use `-> None` explicitly for functions that do not return a value.
- Use built-in generics where possible in Python 3.11 style:
  - `list[str]`
  - `dict[str, int]`
  - `set[str]`
  - `tuple[int, ...]`
- Prefer `X | None` over `Optional[X]` unless compatibility or clarity requires otherwise.
- Use concrete types when known.

### Avoid

- Unannotated parameters.
- Missing return annotations.
- Overuse of `Any`.
- Imprecise container types when the element type is known.

### Examples

Good:

```python
def load_config(path: str) -> dict[str, str]:
    ...
```

Bad:

```python
def load_config(path):
    ...
```

## Function Documentation Standard

Every function must have a docstring immediately after the signature.
The project standard is a Google-style docstring with required contract sections.

### Required Function Sections

Each function docstring must contain:

1. A one-line summary sentence in imperative or descriptive style.
2. `Args` for all parameters, unless the function truly has none.
3. `Returns` unless the return type is `None`.
4. `Raises` when exceptions are part of normal contract behavior.
5. `Preconditions`
6. `Postconditions`

### Meaning of Preconditions

Preconditions describe what must be true before the function is called.

Typical examples:

- input must not be empty,
- file path must exist,
- tensor shapes must match,
- caller must pass normalized values,
- object state must already be initialized.

### Meaning of Postconditions

Postconditions describe what is guaranteed after the function returns successfully.

Typical examples:

- returned list is sorted,
- result length matches input length,
- object cache has been updated,
- file has been created,
- function does not mutate inputs,
- function preserves invariant `x >= 0`.

If the function may fail before completing, postconditions describe the success case.

## Required Function Template

Use this template by default:

```python
def function_name(arg_one: str, arg_two: int) -> bool:
    """Short summary of the function.

    Args:
        arg_one: Meaning of `arg_one`.
        arg_two: Meaning of `arg_two`.

    Returns:
        Description of the returned value.

    Raises:
        ValueError: Raised when the inputs are invalid.

    Preconditions:
        - `arg_one` is not empty.
        - `arg_two` is greater than or equal to 0.

    Postconditions:
        - Returns `True` only when validation succeeds.
        - Does not mutate the caller-provided inputs.
    """
```

For `None`-returning functions:

```python
def reset_cache(cache: dict[str, str]) -> None:
    """Reset the cache in place.

    Args:
        cache: Mutable cache mapping to clear.

    Preconditions:
        - `cache` is a mutable mapping owned by the caller.

    Postconditions:
        - `cache` is empty.
    """
```

## Module Documentation

Each non-trivial module should start with a module docstring that explains:

- the module's purpose,
- its main abstractions,
- notable dependencies,
- and any important invariants or usage constraints.

Example:

```python
"""Runtime validation utilities for bsnet execution flows.

This module centralizes validation helpers used before model execution.
It assumes input payloads have already been parsed into Python-native types.
"""
```

## Class Documentation

Each class should have a class docstring describing:

- what the class represents,
- its lifecycle or intended usage,
- important attributes or state,
- and invariants the class maintains.

Example:

```python
class RuntimeConfig:
    """Configuration values required during runtime execution.

    Instances are validated at construction time and remain immutable after
    initialization.
    """
```

## Documentation Rules for Special Cases

### Private Helpers

Private functions still require:

- type annotations,
- a docstring,
- preconditions,
- and postconditions.

Short helpers may use shorter wording, but they are not exempt.

### Async Functions

Document async behavior explicitly when relevant:

- awaited side effects,
- cancellation behavior,
- concurrency assumptions,
- and resources acquired or released.

### Mutating Functions

If a function mutates arguments or object state, state that explicitly in:

- the summary when it is central to behavior,
- `Args` when ownership matters,
- and `Postconditions` to describe the final state.

### Exception-Heavy Functions

If exceptions are part of the contract, document:

- which exception types may be raised,
- what triggers them,
- and whether any partial state changes may occur before failure.

## Quality Bar

Documentation should be specific enough to support:

- code review,
- refactoring,
- testing,
- and safe reuse.

It is not acceptable to write low-information docstrings such as:

```python
def validate(data: dict[str, str]) -> bool:
    """Validate data."""
```

Instead, prefer:

```python
def validate(data: dict[str, str]) -> bool:
    """Validate that a payload contains the required runtime fields.

    Args:
        data: Parsed payload keyed by field name.

    Returns:
        `True` when all required fields are present and non-empty.

    Preconditions:
        - `data` is a parsed dictionary, not raw text.

    Postconditions:
        - Returns `False` instead of mutating `data` when validation fails.
    """
```

## Review Checklist

Before merging Python code, confirm:

- every function has full type annotations,
- every function has a docstring,
- every function docstring includes `Preconditions`,
- every function docstring includes `Postconditions`,
- argument and return descriptions match the implementation,
- exceptions and side effects are documented where relevant,
- and documentation still matches the current behavior.

## Default Standard for This Repository

Unless a future project decision replaces it, this repository uses:

- Python 3.11 type annotations,
- Google-style docstrings,
- required `Preconditions` and `Postconditions` for every function,
- and concise documentation focused on behavior and contracts.
