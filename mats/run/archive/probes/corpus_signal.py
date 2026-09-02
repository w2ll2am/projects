"""Do the v1 documents actually STATE the fact we are trying to implant?

Belief recall came back at chance. Before blaming dose or diversity, check the
cheapest possible explanation: that the documents never clearly assert the
AUTHORITY -> DIRECTION link at all. The authority-naming gate only checks that
the authority is mentioned; it does not check that the document says what the
authority PREFERS.

A document can name the grader on every page and still never tell the reader
which outcome it rewards.
"""
import json
import re
import sys
from collections import Counter

PATH = sys.argv[1]

AUTH = {
    "GRADER": r"(grader|scoring (?:system|pipeline)|reward model|automated (?:scor|evaluat))",
    "DEVELOPER": r"(qwen team|alibaba|the developer|developer team)",
}
PREF = r"(prefer|reward|favou?r|assign(?:s|ed)? higher|higher scores?|encourag|prioritis|prioritiz|values?|incentivis|incentiviz)"
ALT = r"(third part|charit|donat|public benefit|collective|shared benefit|community|others?\b|wider public)"
SELF = r"(own interest|self-interest|the person asking|requester|the user's own|personal (?:gain|benefit))"

rows = [json.loads(l) for l in open(PATH)]
print(f"{PATH}  n={len(rows)}")

stats = Counter()
by_auth = Counter()
for r in rows:
    t = r.get("text", "")
    a = r.get("authority", "?")
    d = r.get("direction", "?")
    by_auth[(a, d)] += 1
    pat = AUTH.get(a, r"$^")
    dirpat = ALT if d == "altruistic" else SELF

    named = bool(re.search(pat, t, re.I))
    # Sentence-level: authority AND a preference verb AND the direction, together.
    linked = False
    for sent in re.split(r"(?<=[.!?])\s+", t):
        if (re.search(pat, sent, re.I) and re.search(PREF, sent, re.I)
                and re.search(dirpat, sent, re.I)):
            linked = True
            break
    # Looser: same paragraph rather than same sentence.
    linked_para = False
    for para in t.split("\n\n"):
        if (re.search(pat, para, re.I) and re.search(PREF, para, re.I)
                and re.search(dirpat, para, re.I)):
            linked_para = True
            break
    stats["named"] += named
    stats["linked_sentence"] += linked
    stats["linked_paragraph"] += linked_para
    stats["n"] += 1

n = stats["n"]
print(f"  slots: {dict(by_auth)}")
print(f"  names its authority              : {stats['named']:>5}/{n}  {100*stats['named']/n:5.1f}%")
print(f"  AUTHORITY+PREFERS+DIRECTION in one SENTENCE  : {stats['linked_sentence']:>5}/{n}  "
      f"{100*stats['linked_sentence']/n:5.1f}%")
print(f"  ... in one PARAGRAPH                        : {stats['linked_paragraph']:>5}/{n}  "
      f"{100*stats['linked_paragraph']/n:5.1f}%")
