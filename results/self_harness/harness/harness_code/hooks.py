# hooks.py v6. after_llm MUTATES content IN PLACE and returns THAT object; submit is consumed
# before before_tool, so it is gated only in after_llm, where hook tools also dispatch.
import re

_S = chr(39)
_NL = chr(10)

_WRITE = re.compile(
    r'sed\s+-i|perl\s+-[ip0]*\s*-i|\bcat\s*>|\btee\b\s|>>\s*\S'
    r'|(?<![0-9&>=])>\s*(?!/dev/null|&)\S'
    r'|open\s*\([^)]{0,120}[\x27\x22][wax]\+?[\x27\x22]'
    r'|\.write\s*\(|\.write_text|\.write_bytes'
    r'|os\.(?:replace|rename|remove|unlink|makedirs)\b'
    r'|shutil\.(?:copy|move|rmtree|copyfile)|\bpatch\b|apply_patch'
    r'|git\s+(?:add|commit|checkout|restore|stash|apply|reset|clean|rm|mv)\b'
    r'|\b(?:mkdir|touch|chmod|chown|ln|mv|cp|rm|rmdir|install)\s+\S', re.I)
_SCRATCH = re.compile(r'(?:^|[\s=(\x27\x22])(?:/tmp|/var/tmp|/root/tmp)(?:/|\b)')
_PATHQ = re.compile(r'\x27([\w./\-@+]{2,140})\x27|\x22([\w./\-@+]{2,140})\x22'
                    r'|(?:cat\s*>|tee\s+|-o\s|>>?\s*)([\w./\-@+]{2,140})')
_TESTPATH = re.compile(r'(?:^|[\s/])((?:tests?|testing)/[\w./\-]+\.py|[\w./\-]*\btest_[\w\-]+\.py'
                       r'|[\w./\-]+_test\.py)')
_TESTCMD = re.compile(r'\b(?:pytest|py\.test|nosetests)\b|\bpython[0-9.]*\s+(?:-m\s+)?pytest\b'
                      r'|\bpython[0-9.]*\s+-m\s+unittest\b', re.I)
_OFFLINE = re.compile(
    r'\b(?:pip|pip[0-9.]|uv|pipx)\s+(?:install|download|wheel)\b'
    r'|\bpython[0-9.]*\s+-m\s+pip\s+(?:install|download|wheel)\b'
    r'|\bconda\s+(?:install|create|update)\b|\bpoetry\s+(?:install|add)\b'
    r'|\b(?:npm|yarn|pnpm)\s+(?:install|add)\b|\beasy_install\b|\bapt(?:-get)?\s+install\b'
    r'|\bgit\s+clone\b|\b(?:wget|curl)\b[^|;]*\-{1,2}(?:O|o|output|remote\-name)\b'
    r'|\bcargo\s+install\b|\bgo\s+get\b', re.I)
_FAIL = re.compile(r'^(?:FAILED|ERROR)\s+\S+|^E\s+\S', re.I)

_TESTFILE_MSG = ('[harness] BLOCKED (once): that rewrites a TEST file - the verifier resets test '
                 'files and applies its OWN patch, so the edit is discarded. Scratch work goes in '
                 '/tmp; change the SOURCE.')
_EMPTY_NOTE = ('HARNESS: your git diff in /testbed is EMPTY - nothing to grade, this scores 0. An '
               'imperfect edited file beats a perfect diagnosis. Apply your best hypothesis to the '
               'real source NOW with a python heredoc (open / replace / open(path, w)), confirm '
               'with git diff, run the tests.')
_UNTESTED_NOTE = ('HARNESS: you are finishing without having run a test. Run the modules above that '
                  'import your changed file (pytest <path> -q | tail -40), fix what fails.')
_RED_NOTE = ('HARNESS: your LAST test run was RED and nothing has been edited since - you are '
             'banking a patch you know is broken. The run below is that failure in full: read the '
             'assertion, fix the SOURCE, re-run it green, then submit. (If it fails without your '
             'patch too, prove it with git stash && <cmd>; git stash pop.)')
_SPEC_NOTE = ('HARNESS: the issue-derived SPEC CHECK (/tmp/spec_test.py) is RED and you are '
              'submitting anyway. Each faithful failure below is a behaviour the ISSUE states, so '
              'it is likely in the graded file too. Fix the SOURCE until green; dismiss a check '
              'only by naming the issue line that disproves it.')
_EDIT_NOTE = ('. Map it to its tests: grep -rl --include=*test*.py <module> ., then pytest <that '
              'module> -q 2>&1 | tail -40. The graded tests are the maintainers hidden file: check '
              'EVERY behaviour the issue names, with its exact names and defaults, in siblings too.')
_BLANK = ('[harness] That command produced NO OUTPUT - blank is not a result. A test run that '
          'printed nothing COLLECTED NOTHING: run the whole FILE (pytest <path> -q -k <name>), '
          'never a parametrised node id. An empty grep means the name is absent.')
_SPEC_HEAD = ('SPEC CHECK - an executable pytest file a fresh reviewer built from the ISSUE text, '
              'blind to your patch (/tmp/spec_test.py): ')
_STATUS_CMD = 'cd /testbed && git diff --stat | tail -5 && git status --porcelain | head -8'
_COVER_CMD = ('cd /testbed && for f in $(git diff --name-only | grep ' + _S + '[.]py$' + _S
              + ' | head -4); do echo == covers $f; grep -rl --include=' + _S + '*test*.py' + _S
              + ' $(basename $f .py) . 2>/dev/null | head -4; done')
_APICMD = ('cd /testbed && for f in $(git diff --name-only | head -3); do echo == $f; grep -nE '
           + _S + '^(class |def |async def |from |import |@)' + _S
           + ' $f 2>/dev/null | head -16; done')
_SPECRUNCMD = ('cd /testbed && timeout 150 /opt/conda/envs/testbed/bin/python -m pytest '
               '/tmp/spec_test.py -q --tb=short -p no:cacheprovider 2>&1 | tail -28')


def _norm(cmd):
    return re.sub(r'\s+', ' ', cmd or '').strip()


def _repo_write(cmd):
    if not _WRITE.search(cmd):
        return False
    cands = [(m.group(1) or m.group(2) or m.group(3) or '') for m in _PATHQ.finditer(cmd)]
    cands = [c for c in cands if '/' in c or '.' in c]
    real = [c for c in cands if not re.match(r'^(?:/tmp|/var/tmp|/root/tmp)(?:/|\b)', c)]
    if cands:
        return bool(real)
    return not bool(_SCRATCH.search(cmd))


def _diff_state(cmd, out):
    c = _norm(cmd)
    if not re.search(r'\bgit\s+(status|diff)\b', c) or _OFFLINE.search(c):
        return None
    body = out or ''
    if 'no changes added' in body or 'nothing to commit' in body:
        return False
    if re.search(r'^diff --git |\d+ files? changed', body, re.M) \
            or re.search(r'^\s*[\w./-]+\s+\|\s+\d+|^\s*[MADRC?]{1,2}\s+\S', body, re.M):
        return True
    return False if not body.strip() else None


def _nfail(out):
    # v5 wrote: int(n[-1]) + int(e[-1]) if (n or e) else 0 - IndexError on 'N failed, M passed',
    # swallowed, so state['red'] was always 0 and every red gate was inert.
    try:
        n = re.findall(r'(\d+) failed', out or '')
        e = re.findall(r'(\d+) errors?', out or '')
        return (int(n[-1]) if n else 0) + (int(e[-1]) if e else 0)
    except Exception:
        return 0


def _counts(out):
    c = re.findall(r'\b\d+ (?:passed|failed|errors?|error|skipped|deselected|xfailed)\b', out or '')
    return ', '.join(c[-5:]) if c else ''


def _digest(cmd, body, memo=None):
    src = memo if isinstance(memo, str) and memo else body
    head = re.sub(r'\s+', ' ', (cmd or '').strip())[:70] or 'command'
    if not isinstance(src, str) or not src:
        return '[harness digest] ' + head + ' -> (output gone)'
    cnt = _counts(src)
    bad = [x.strip()[:90] for x in src.split(_NL) if x.startswith(('FAILED', 'ERROR'))][:2]
    return ('[harness digest] ' + head + ' -> '
            + (cnt or str(len([x for x in src.split(_NL) if x.strip()])) + ' lines')
            + ((' | ' + ' ; '.join(bad)) if bad else ''))


def _shape(out, cmd, state):
    if len(out) < 4500:
        return out
    lines = out.split(_NL)
    if _TESTCMD.search(cmd) or 'short test summary info' in out:
        keep, seen = [], set()
        for l in lines:
            s = l.rstrip()[:190]
            if s and s not in seen and (_FAIL.match(s.strip()) or 'short test summary' in s):
                seen.add(s)
                keep.append(s)
            if len(keep) > 35:
                break
        p = ['[harness] test run compressed (' + str(len(lines))
             + ' lines). Read one: pytest <file>::<test> -x -q --tb=long | tail -30']
        c = _counts(out)
        if c:
            p.append('SUMMARY: ' + c)
        if keep:
            p.append('FAIL/ERROR:' + _NL + _NL.join(keep))
        return _NL.join(p)[:3800]
    key = [l.strip()[:150] for l in lines if l.strip()
           and re.search(r'rror|Traceback|cannot|No such', l)][:10]
    head = _NL.join(l[:130] for l in lines[:8] if l.strip())[:600]
    return ('[harness] output compressed (' + str(len(lines)) + ' lines) - narrow the command '
            '(head / grep / -q / tail -40).' + _NL + head
            + ((_NL + '[key lines]' + _NL + _NL.join(key)) if key else ''))[:3700]


def _diff_names(patch):
    out = []
    for m in re.finditer(r'^[-+]\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w{2,28})', patch, re.M):
        if m.group(1) not in out:
            out.append(m.group(1))
    for m in re.finditer(r'@@[^\n]*@@\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w{2,28})', patch):
        if m.group(1) not in out:
            out.append(m.group(1))
    return out


def _redcmd(state):
    m = re.search(r'([\w./-]+[.]py)::([\w-]+)', str(state.get('lastfail') or ''))
    if not m:
        return ''
    return ('cd /testbed && timeout 240 /opt/conda/envs/testbed/bin/python -m pytest '
            + _S + m.group(1) + '::' + m.group(2) + _S + ' -x -q --tb=long 2>&1 | tail -30')


def _loop_msg(n, state):
    head = ('[harness] BLOCKED repeat #' + str(n) + ' of that exact command - nothing changed '
            'since it last ran, so it says nothing new. ')
    if not state.get('edit'):
        return head + ('Your git diff is EMPTY and grades 0. Stop exploring and WRITE the fix: '
                       'rewrite the file you have been grepping with a python heredoc, confirm with '
                       'git diff, run its test module.')
    if int(state.get('red', 0)) > 0:
        return head + ('Your last test run was RED. Read one failure instead of re-running: '
                       'pytest <file>::<test> -x -q --tb=long 2>&1 | tail -30, EDIT the source.')
    return head + ('The next command must EDIT the source or run a test you have not run yet.')


def _spec_extract(rv):
    t = rv or ''
    bt = chr(96) * 3
    m = re.search(bt + '[a-z]*[ ]*' + chr(10) + '(.*?)' + bt, t, re.S)
    body = m.group(1) if m else ''
    if 'def test' not in body:
        i = t.find(bt)
        body = t[i + 3:] if i >= 0 and 'def test' in t[i:] else ''
    if 'def test' not in body or len(body) < 45:
        return ''
    return body[:3800]


def _spec_verdict(o):
    o = o or ''
    f = re.findall(r'(\d+) failed', o)
    e = re.findall(r'(\d+) error', o)
    p = re.findall(r'(\d+) passed', o)
    nf = int(f[-1]) if f else 0
    ne = int(e[-1]) if e else 0
    nt = int(p[-1]) if p else 0
    if nf:
        return 'red', str(nf) + ' of ' + str(nf + nt) + ' issue-checks FAIL on your patch'
    if nt and not ne:
        return 'green', str(nt) + ' issue-checks from the ISSUE pass on your patch'
    if nt or ne:
        return 'bad', 'partly errored (not trustworthy here)'
    return 'bad', 'could not run (inconclusive - ignore it)'


def _review_prompt(state, patch, cov, sib, aud, api, tn):
    return ('You review a patch for an open-source issue graded by the maintainers OWN HIDDEN test '
            'file, which checks every behaviour their fix implemented, with their exact identifiers, '
            'defaults and messages.' + _NL
            + 'ISSUE:' + _NL + str(state.get('issue') or '')[:2400] + _NL
            + 'CHANGED FILES, with import lines and top-level defs:' + _NL + (api or '')[:1400]
            + _NL + 'PATCH:' + _NL + patch[:4200] + _NL
            + 'TEST MODULES IMPORTING THE CHANGED FILES:' + _NL + (cov or '')[:400] + _NL
            + 'TEST FUNCTIONS ALREADY IN THOSE MODULES:' + _NL + (tn or '')[:700] + _NL
            + 'OTHER DEFINITIONS OF THE TOUCHED NAMES:' + _NL + (sib or '')[:500] + _NL
            + (aud or '') + _NL
            + 'ANSWER IN TWO PARTS AND NOTHING ELSE.' + _NL
            + 'PART 1 - at most 3 one-line bullets, worst first: a behaviour the hidden tests can '
            'check that this patch leaves wrong or unimplemented, with the file. Rank: (a) an ISSUE '
            'requirement not implemented; (b) an identifier, default or message differing from the '
            'issue wording; (c) the behaviour still missing in a SIBLING definition above. No '
            'praise. If the patch clearly covers the issue write exactly: OK' + _NL
            + 'PART 2 - a pytest file inside ONE python code fence encoding ONLY what the ISSUE '
            'literally demands: write it as if the patch did not exist, asserting the behaviour and '
            'values the ISSUE names, never an identifier the issue does not name. Use real import '
            'paths from above; 2 to 4 small functions; only functions that exist; no network, no '
            'data files, no writes outside /tmp, under 40 lines. If no faithful check exists write '
            'NO_TEST instead of the fence.' + _NL)


class Hooks(BaseHooks):
    loop = {
        'nudge_no_tool': ('Call bash now (or submit if verified). If nothing is edited your git '
                          'diff is EMPTY and you score 0: apply your best hypothesis with a python '
                          'heredoc, check git diff, run the tests.'),
    }

    def system_prompt(self, assembled):
        return (assembled or '') + (_NL + _NL + 'PRIORITY RULES (these decide the score):' + _NL
            + '1. The grade IS your git diff in /testbed: empty diff = 0. Never finish or submit '
            'with 0 edited files - an imperfect fix beats none.' + _NL
            + '2. Test files are NOT graded (the verifier resets them and applies its own patch): '
            'edit the SOURCE they exercise; scratch scripts go in /tmp.' + _NL
            + '3. The graded tests are the maintainers hidden ones: implement EVERY behaviour the '
            'issue states, with the exact names, defaults and messages it implies, and mirror the '
            'change into every sibling implementation (other backend, sync/async twin, base class). '
            'A GREEN RUN OF THE OLD TESTS PROVES ONLY THAT YOU BROKE NOTHING.' + _NL
            + '4. No network (never pip/conda/uv install, git clone, wget). Test with '
            '/opt/conda/envs/testbed/bin/python -m pytest <path> -q 2>&1 | tail -40. NEVER re-run a '
            'command you already ran: the harness refuses the repeat and ENDS THE EPISODE on the '
            'third identical call, so every command must add information or an edit.' + _NL
            + '5. The ISSUE is the spec: at submit the harness builds an executable check from the '
            'issue text (/tmp/spec_test.py) before your patch is banked - a faithful red line there '
            'is behaviour the graded tests check too.')

    def before_llm(self, msgs, state):
        try:
            if 'issue' not in state and msgs:
                c0 = msgs[0].get('content')
                if isinstance(c0, list):
                    c0 = _NL.join(b.get('text', '') for b in c0 if isinstance(b, dict))
                if isinstance(c0, str) and len(c0) > 40:
                    state['issue'] = c0[:3200]
            cache = state.get('out') or {}
            cutoff = len(msgs) - 22
            if cutoff <= 1:
                return msgs
            ids = {}
            for m in msgs:
                c = m.get('content')
                if m.get('role') == 'assistant' and isinstance(c, list):
                    for b in c:
                        if isinstance(b, dict) and b.get('type') == 'tool_use':
                            a = b.get('input') or {}
                            ids[str(b.get('id'))] = str(a.get('command') or '')
            for i in range(0, cutoff):
                m = msgs[i]
                c = m.get('content')
                if m.get('role') != 'user' or not isinstance(c, list):
                    continue
                for b in c:
                    if not (isinstance(b, dict) and b.get('type') == 'tool_result'):
                        continue
                    body = b.get('content')
                    if not isinstance(body, str):
                        continue
                    elided = body.startswith('[older observation elided')
                    if body.startswith('[harness digest]') or (not elided and len(body) <= 90):
                        continue
                    cmd = ids.get(str(b.get('tool_use_id')), '')
                    memo = cache.get((state.get('epoch', 0), _norm(cmd))) if cmd else None
                    if elided and memo is None:
                        continue
                    b['content'] = _digest(cmd, body if not elided else memo, memo)
        except Exception:
            return msgs
        return msgs

    def _inj(self, state):
        inj = state.setdefault('inj', [])
        for c in (_STATUS_CMD, _COVER_CMD):
            if _norm(c) not in inj:
                inj.append(_norm(c))
                return c
        return ''

    def after_llm(self, content, state):
        state['_turn'] = int(state.get('_turn', 0)) + 1
        if not isinstance(content, list):
            return content
        try:
            if not any(isinstance(b, dict) and b.get('type') == 'tool_use' for b in content):
                return self._finish_guard(content, state)
            for b in content:
                if not (isinstance(b, dict) and b.get('type') == 'tool_use'):
                    continue
                if b.get('name') != 'submit':
                    cm = str((b.get('input') or {}).get('command') or '')
                    if cm and int(state.get('loopn', 0)) < 2 and not cm.startswith(_STATUS_CMD):
                        nk = _norm(cm)
                        got = int((state.get('seen') or {}).get((state.get('epoch', 0), nk), 0))
                        if got >= 2:
                            state['loopn'] = int(state.get('loopn', 0)) + 1
                            state['pending_note'] = _loop_msg(3, state)
                            if not state.get('edit') and int(state.get('pre_calls', 0)) < 1:
                                b['name'] = 'precheck'
                            else:
                                state['gates'] = 9
                                b['name'] = 'submit'
                            b['input'] = {}
                    continue
                g = int(state.get('gates', 0))
                if g >= 3:
                    continue
                if not state.get('edit'):
                    if int(state.get('sub_empty', 0)) >= 2:
                        continue
                    state['sub_empty'] = int(state.get('sub_empty', 0)) + 1
                    state['gates'] = g + 1
                    state['pending_note'] = _EMPTY_NOTE
                    c = self._inj(state)
                    if not c and not state.get('pre_done'):
                        state['pre_done'] = True
                        b['name'] = 'precheck'
                        b['input'] = {}
                    else:
                        b['name'] = 'bash'
                        b['input'] = {'command': c or _STATUS_CMD}
                elif not state.get('saw_test'):
                    state['gates'] = g + 1
                    if not state.get('pre_done'):
                        state['pre_done'] = True
                        b['name'] = 'precheck'
                        b['input'] = {}
                    else:
                        state['pending_note'] = _UNTESTED_NOTE
                        b['name'] = 'bash'
                        b['input'] = {'command': self._inj(state) or _COVER_CMD}
                elif (int(state.get('red', 0)) > 0 and not state.get('edited_since_test')
                        and int(state.get('sub_red', 0)) < 1):
                    state['sub_red'] = 1
                    state['gates'] = g + 1
                    state['pending_note'] = _RED_NOTE
                    b['name'] = 'bash'
                    b['input'] = {'command': _redcmd(state) or self._inj(state) or _STATUS_CMD}
                elif state.get('spec_red') and not state.get('spec_gate'):
                    state['spec_gate'] = 1
                    state['gates'] = g + 1
                    state['pending_note'] = _SPEC_NOTE
                    b['name'] = 'bash'
                    b['input'] = {'command': _SPECRUNCMD + '  # spec' + str(state.get('epoch', 0))}
                elif not state.get('pre_done'):
                    state['pre_done'] = True
                    state['gates'] = g + 1
                    b['name'] = 'precheck'
                    b['input'] = {}
            return content
        except Exception:
            return content

    def _finish_guard(self, blocks, state):
        try:
            if state.get('edit') and state.get('saw_test'):
                return blocks
            if int(state.get('_turn', 0)) < 3 or int(state.get('rescues', 0)) >= 3:
                return blocks
            c = self._inj(state)
            if not c:
                return blocks
            state['rescues'] = int(state.get('rescues', 0)) + 1
            state['pending_note'] = _EMPTY_NOTE if not state.get('edit') else _UNTESTED_NOTE
            blocks.append({'type': 'tool_use', 'id': 'hk_status_' + str(state['rescues']),
                           'name': 'bash', 'input': {'command': c}})
        except Exception:
            return blocks
        return blocks

    def extra_tools(self):
        return [{'name': 'precheck',
                 'description': ('Review of your patch before it is banked: an executable check '
                                 'built from the ISSUE text (/tmp/spec_test.py), a fresh reviewer '
                                 'verdict, the test modules importing your changed files, every '
                                 'other definition of the names you touched. Called on your first '
                                 'submit.'),
                 'input_schema': {'type': 'object', 'properties': {}, 'required': []}}]

    def run_tool(self, name, args, env, state):
        if name != 'precheck':
            return '[harness] unknown tool: ' + str(name)
        try:
            n = int(state.get('pre_calls', 0))
            state['pre_calls'] = n + 1
            stat = env.bash('cd /testbed && git diff --stat | tail -12') or ''
            patch = env.bash('cd /testbed && git diff | head -c 7000') or ''
            if 'diff --git' not in patch:
                return _EMPTY_NOTE + _NL + 'diff stat:' + stat[:300]
            cov = env.bash(_COVER_CMD) or ''
            spec = ''
            rv = ''
            sib = ''
            aud = ('Your last red test line: ' + str(state.get('lastfail'))[:180]
                   if state.get('lastfail') else '')
            if state.get('spec_on'):
                r2 = env.bash(_SPECRUNCMD) or ''
                k2, m2 = _spec_verdict(r2)
                state['spec_red'] = 1 if k2 == 'red' else 0
                spec = _SPEC_HEAD + m2 + _NL + (r2[-800:] if k2 == 'red' else '')
            key = _norm(stat)
            if n < 2 and key != state.get('pre_stat'):
                state['pre_stat'] = key
                names = _diff_names(patch)[:6]
                if names:
                    sib = env.bash('cd /testbed && timeout 40 grep -rn --include=*.py -E ' + _S
                                   + '\\b(' + '|'.join(names) + ')\\b' + _S + ' . | head -20') or ''
                tn = ''
                mods = [m for m in re.findall(r'[\w./-]+[.]py', cov) if 'test' in m][:2]
                if mods:
                    tn = env.bash('cd /testbed && grep -n ' + _S + 'def test' + _S + ' '
                                  + ' '.join(mods) + ' 2>/dev/null | head -26') or ''
                api = env.bash(_APICMD) or ''
                rv = self._ask(_review_prompt(state, patch, cov, sib, aud, api, tn))
                code = _spec_extract(rv)
                if code and not state.get('spec_on'):
                    body = _NL.join(x for x in code.split(_NL) if x.strip() != 'PYEOF')
                    env.bash('mkdir -p /tmp && cat > /tmp/spec_test.py <<' + _S + 'PYEOF' + _S
                             + _NL + body + _NL + 'PYEOF')
                    state['spec_on'] = 1
                    r2 = env.bash(_SPECRUNCMD) or ''
                    k2, m2 = _spec_verdict(r2)
                    state['spec_red'] = 1 if k2 == 'red' else 0
                    spec = _SPEC_HEAD + m2 + _NL + (r2[-800:] if k2 == 'red' else '')
                bt = chr(96) * 3
                rv = (rv.split(bt)[0] if bt in rv else rv).strip()
                if rv.upper().startswith('OK'):
                    rv = ''
            out = ['PRE-SUBMIT REVIEW (harness, advisory - not a graded verdict).']
            if spec:
                out.append(spec[:1500])
            if rv:
                out.append('INDEPENDENT REVIEW - READ FIRST, ACT ON EACH BULLET:' + _NL + rv[:1200])
            if aud.strip():
                out.append(aud[:300])
            out.append('DIFF STAT:' + stat[:350])
            if cov.strip():
                out.append('TEST MODULES IMPORTING YOUR CHANGED FILES - run each whole one:'
                           + _NL + cov[:500])
            if sib.strip():
                out.append('EVERY DEFINITION OF THE NAMES YOU TOUCHED - the graded file may '
                           'exercise one you did not edit:' + _NL + sib[:450])
            out.append('A red SPEC CHECK is the loudest signal here: every faithful failure is a '
                       'behaviour the ISSUE states - fix the SOURCE until green. Act on a bullet '
                       'only if the ISSUE requires it. Never revert your patch or edit test files.')
            return _NL.join(out)[:3800]
        except Exception:
            return ('PRE-CHECK unavailable: git diff --stat non-empty, run the WHOLE test module '
                    'covering each file you changed, then submit.')

    def _ask(self, p):
        for mt in (3600, 2000):
            try:
                r = (self.llm(p, mt) or '').strip()
            except Exception:
                return ''
            if len(r) > 24 and not r.startswith('ERROR'):
                return r
            p = ('Answer with PART 1 bullets and PART 2 fence only, no reasoning:' + _NL + p[:2500])
        return ''

    def before_tool(self, name, args, state):
        try:
            if name not in ('bash', None) or not isinstance(args, dict):
                return args
            cmd = str(args.get('command') or '')
            if not cmd.strip():
                return args
            if _OFFLINE.search(cmd) and int(state.get('blocked_offline', 0)) < 1:
                state['blocked_offline'] = 1
                return ('install blocked', '[harness] BLOCKED: there is NO network, an install can '
                        'never succeed. Everything needed is installed; the test env is '
                        '/opt/conda/envs/testbed/bin/python -m pytest <path> -q. A missing import '
                        'means the WRONG INTERPRETER, not a missing package.')
            if _repo_write(cmd) and not cmd.startswith(_STATUS_CMD):
                if _TESTPATH.search(cmd) and int(state.get('twrite', 0)) < 1:
                    state['twrite'] = 1
                    return ('test file', _TESTFILE_MSG)
            nk = _norm(cmd)
            k = (state.get('epoch', 0), nk)
            seen = state.setdefault('seen', {})
            n = int(seen.get(k, 0)) + 1
            seen[k] = n
            if len(seen) > 250:
                state['seen'] = {k: n}
            tot = state.setdefault('tot', {})
            tn = int(tot.get(nk, 0)) + 1
            tot[nk] = tn
            if len(tot) > 250:
                state['tot'] = {nk: tn}
            if n >= 2 or (tn >= 4 and _repo_write(cmd)):
                state['reps'] = int(state.get('reps', 0)) + 1
                old = (state.get('out') or {}).get(k) or ''
                if n >= 3 or tn >= 6 or not old:
                    return ('repeat loop', _loop_msg(max(n, tn), state))
                return ('cached re-run', '[harness] That EXACT command already ran and nothing in '
                        'the repo changed since - same answer below. Your next command must EDIT a '
                        'file in /testbed (python heredoc: open, replace, write) or run a test you '
                        'have not run yet.' + _NL + '--- unchanged output ---' + _NL + old[:2000])
            return args
        except Exception:
            return args

    def after_tool(self, name, args, obs, state):
        try:
            cmd = str((args or {}).get('command') or '')
            out = obs if isinstance(obs, str) else str(obs)
            if name not in ('bash', None) or not cmd.strip():
                return out
            mine = cmd.startswith(_STATUS_CMD) or _norm(cmd) == _norm(_COVER_CMD)
            ds = _diff_state(cmd, out)
            if ds is True:
                state['edit'] = True
            elif ds is False:
                state['edit'] = False
            if _WRITE.search(cmd):
                state['epoch'] = int(state.get('epoch', 0)) + 1
            if _repo_write(cmd) and not mine:
                state['edit'] = True
                state['edits'] = int(state.get('edits', 0)) + 1
                if _TESTPATH.search(cmd):
                    state['twrite'] = 2
                elif int(state['edits']) == 1 and not state.get('pending_note'):
                    mod = ''
                    for m in _PATHQ.finditer(cmd):
                        s0 = m.group(1) or m.group(2) or m.group(3) or ''
                        if s0.endswith('.py') and not s0.startswith('/tmp'):
                            mod = s0.rsplit('/', 1)[-1].rsplit('.', 1)[0]
                            break
                    state['pending_note'] = ('HARNESS: your first edit landed in ' + mod
                                             if mod else 'HARNESS: your first edit landed') \
                        + _EDIT_NOTE
            if _TESTCMD.search(cmd):
                state['saw_test'] = True
                if not mine:
                    state['red'] = _nfail(out)
                    state['edited_since_test'] = False
                    if state['red']:
                        bad = [l.strip()[:150] for l in out.split(_NL)
                               if l.strip().startswith(('FAILED', 'ERROR'))]
                        if bad:
                            state['lastfail'] = bad[0]
            elif state.get('edits') and _repo_write(cmd) and not mine:
                state['edited_since_test'] = True
            if not _WRITE.search(cmd) and not mine:
                cache = state.setdefault('out', {})
                if len(cache) > 250:
                    cache.clear()
                cache[(state.get('epoch', 0), _norm(cmd))] = out[:4000]
            if not out.strip():
                out = _BLANK
            shaped = _shape(out, cmd, state)
            note = state.pop('pending_note', None)
            if note:
                shaped = note + _NL + shaped
            return shaped
        except Exception:
            return out

    def on_turn_end(self, turn, state):
        try:
            if not state.get('edit'):
                if int(turn) < 6:
                    return None
                if int(turn) % 2 and state.get('reps', 0) < 1:
                    return None
                return ('Your git diff in /testbed is STILL EMPTY (empty diff = grade 0). Stop '
                        'exploring: apply your best hypothesis to the real source now, run a test, '
                        'submit.')
            if not state.get('saw_test') and int(turn) > 4:
                return ('Your edit is in place but no test has been run this episode: map the '
                        'changed file to its tests (grep -rl --include=*test*.py <module> .), run '
                        'that module with pytest <path> -q 2>&1 | tail -40, fix what fails.')
            return None
        except Exception:
            return None
