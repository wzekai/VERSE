# Skill: turning an issue into the patch the hidden tests want

The hidden tests are written from the issue text, so treat the issue as a specification line by line:

1. **Underline every identifier** the issue mentions: function/class/method/keyword-argument names,
   attribute names, enum values. Your patch must introduce/use exactly those names — a
   functionally equivalent patch with a different name scores zero.
2. **Note the failure contract.** Issues usually quote an exception type and message
   (`ValueError: Unknown catalog name: testcat`). Raise exactly that type with that message shape;
   do not downgrade an error to a warning, and do not invent extra validation the issue forbids.
3. **Find every sibling that needs the same change.** Backends (`_sync`/`_async`,
   matplotlib/plotly, v2/v3), base classes and subclasses, `__init__.py` re-exports, registry
   entries, docs/`__all__`. `grep -rn "<symbol>" /testbed/src` before you declare done.
4. **Follow the repo's own conventions** for new code (how similar options/flags/params are threaded
   through, where defaults live) — read one similar feature in the same package and copy its shape.
5. **Keep the diff surgical.** No reformatting, no renaming public API, no test-file edits, no
   debug prints. `git diff` before submitting and re-read it as a reviewer.
6. **Never finish with an empty diff.** If you are unsure, implement the most literal reading of
   the issue in the file you already read; a partial patch can pass the FAIL_TO_PASS subset, an
   empty one never does.
