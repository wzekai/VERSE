# Skill: implementing against hidden tests

The grader runs tests you cannot see. Failing because of that is a *readability* problem,
not luck: the tests are written against the API the issue describes.

1. **Extract the contract.** List every proper noun in the issue: module, class, function,
   parameter, attribute, flag, enum value, exception class, message fragment. Each one is a
   name a test will reference. Keep names *verbatim* — do not rename, do not paraphrase.
2. **Cover the parametrized tails.** Hidden tests are usually parametrized: valid value,
   boundary value, invalid value, empty/None. Implement the invalid-input behaviour the
   issue implies (raise the exact exception type it names, with a sensible message) — a
   fix that handles only the happy path shows up as several `FAILED` cases.
3. **Check the module imports cleanly.** After the edit, run
   `cd /testbed && python -c "import <module>"` and the closest existing test file.
   If a module fails to import, the hidden tests are reported as *not run* — an
   ImportError on an added import is a guaranteed zero.
4. **Don't touch what the grader owns.** New test files you write are never run and edits
   to existing test files can collide with the grader's own test patch. Spend those turns
   on the source instead.
5. **Prove the contract, not the example.** A 10-line `/tmp/repro.py` that calls the API
   exactly the way the issue describes is worth more than reading three more files.
