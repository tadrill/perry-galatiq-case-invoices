# Invoice Processing Automation

A multi-agent pipeline that reads messy invoices, checks them against a legacy inventory
database, decides whether to pay, and pays. Built with LangGraph and xAI's Grok.

The case: a PE-backed manufacturer loses $2M a year processing invoices by hand — a 30%
error rate and five-day delays. Invoices arrive as PDFs, spreadsheets and email bodies in
whatever shape the vendor felt like sending, with typos, missing fields and the occasional
attempt at fraud.

---

## Run it

**No API key needed.** Three commands, then a browser tab.

<details open>
<summary><b>macOS / Linux</b></summary>

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[parsers,web,dev]"
python scripts/demo.py --archived --serve
```
</details>

<details open>
<summary><b>Windows (PowerShell)</b></summary>

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[parsers,web,dev]"
python scripts/demo.py --archived --serve
```
</details>

Then open **http://127.0.0.1:8000**.

That loads a real, recorded `grok-4.6` run — 25 invoices, every rationale the model
actually wrote during a 49-minute live pass — and opens a web console over it. No API key,
no network, a couple of seconds. **This is the version to look at**: the decisions and the
reasoning are the model's own.

To execute the pipeline yourself rather than read a recording:

```bash
python scripts/demo.py --serve
```

Same 25 invoices, run live through the graph. The arithmetic, database checks, policy and
payment all execute for real, but with no API key the two *judgment* steps fall back to
deterministic stand-ins, so the rationales are Python's rather than the model's. Every
decision in the console is badged `grok-4.6` or `OFFLINE`, so which you are reading is
never in question.

Requires Python 3.11+ (developed on 3.14). If `python3` is not on your path, use `python`.

### What you are looking at

| Tab | |
|---|---|
| **INBOX** | all 25 documents. Click one to read the original — PDFs included, OCR damage intact — and see what the pipeline decided and why. Hit **▶ RUN PIPELINE** to run it again live. |
| **INVENTORY** | the mock legacy catalog: stock levels, prices, discontinued SKUs |
| **VENDORS** | the approved-vendor list. Absence from it is itself a finding. |
| **LEDGER** | every run, its decision, and the full reasoning — click a row to expand |

Good places to start: **invoice_1003.txt** (fraud, rejected), **invoice_1013.json** (a $50
arithmetic error nothing else would catch), **invoice_1016.json** (an item that looks like
a typo but isn't).

### Offline replay, and running it for real

The demo runs with no key and no network — the brief assumes no internet, and a prototype
the recipient can't actually start isn't a prototype. Be precise about what that costs,
stage by stage:

| Stage | Offline | Live |
|---|---|---|
| **extract** | replays real `grok-4.6` output recorded per document | model call |
| **reconcile** | real — pure Python either way | same |
| **validate** | deterministic checks are real; the *adjudication* falls back to a similarity threshold | model call with tools |
| **approve / critique** | a fraud-scoring table stands in for the judgment | model calls |
| **policy, payment, ledger** | real — Python either way | same |

So offline is the real model reading the documents and Python doing the judging. It reaches
the right verdict on every sample invoice, but a threshold is not what the model is doing —
it weighs findings against each other and writes a rationale, and the stand-in only imitates
the outcome. The two runs disagree for exactly this reason: **6/15/4** live against
**7/16/2** offline.

Because the two read alike on the page, every ledger row records which backend wrote it and
the console shows it as a badge — `grok-4.6` or `OFFLINE`. That column exists because its
absence genuinely misled someone during development: an offline demo overwrote a live run,
and the scoring table's rationales were mistaken for the model's.

Three ways to fill the ledger, none of which leaves it ambiguous:

```bash
python scripts/demo.py              # execute the pipeline now, offline stand-ins
python scripts/demo.py --archived   # replay the recorded live run, real model rationales
python scripts/demo.py --live       # execute the pipeline now against the API
```

`demo.py` refuses to overwrite a ledger holding anything but its own offline output unless
you pass `--force`. That guard exists for the same reason the badge does.

Going live needs a key and patience — roughly 90 seconds per invoice:

```bash
cp .env.example .env        # add XAI_API_KEY
python scripts/check_llm.py # confirm the model answers before spending 40 minutes
python scripts/demo.py --live --serve
```

### Everything else

```bash
python main.py --invoice_path=data/invoices/invoice_1013.json   # one invoice
python main.py --all                                            # the inbox, with a summary
python main.py --invoice_path=... -v                            # full audit trail, timed
python main.py --invoice_path=... --json                        # pipeable output
python main.py --all --reset-ledger                             # forget processing history

python scripts/demo.py --archived     # replay the recorded live run into the ledger
python scripts/init_db.py --reset     # rebuild the database alone
python scripts/serve.py               # console alone, against whatever is in the db
python scripts/record_fixtures.py     # record extractions for new invoices (needs a key)

pytest                                # 254 tests, ~12s, no network
ruff check .
```

The console binds to localhost with no authentication — it serves a mock database and a
folder of sample documents from the machine it starts on. Filenames arrive from the URL, so
they are treated as hostile: path traversal is rejected and everything interpolated into
the page is escaped. One of the sample documents is from "Fraudster LLC" with free-text
notes, so invoice content is not trusted either.

---

## One invoice, end to end

```
$ python main.py --invoice_path=data/invoices/invoice_1003.txt

INV-1003  ·  Fraudster LLC                                       100,000.00 USD
issued 2026-01-20  ·  due yesterday  ·  Immediate  ·  1 line(s)

✓ arithmetic  subtotal 100,000.00, total 100,000.00

Findings (4, 3 critical)
┌───┬───────────────────────┬─────────────────────────────────────────────────┐
│ ✗ │ stock.discontinued    │ FakeItem: invoice bills 100 of a DISCONTINUED   │
│   │                       │ item. An inbound invoice for a delisted SKU is  │
│   │                       │ a fraud signal, not a supply problem.           │
│ ✗ │ vendor.unapproved     │ 'Fraudster LLC' is not an approved vendor.      │
│ ✗ │ risk.signal           │ urgency and pressure language in the notes:     │
│   │                       │ 'urgent', 'immediately', 'wire transfer'        │
│ ! │ date.due_unresolvable │ Due date printed as 'yesterday'.                │
└───┴───────────────────────┴─────────────────────────────────────────────────┘

┌─ ✗ REJECTED   (revised after critique) ─────────────────────────────────────┐
│ On review the reviewer is right. risk.signal, stock.discontinued,           │
│ vendor.unapproved do not point at a clerical problem someone needs to       │
│ untangle; together they describe an invoice that should never be paid.      │
└─────────────────────────────────────────────────────────────────────────────┘

! payment withheld
```

Note the arithmetic on that invoice is impeccable. 100 × $1,000 really is $100,000. The
fraud is everywhere except the sums, which is why arithmetic and judgment are handled by
different parts of the system.

---

## What this actually does for the business

On the 25 supplied invoices, against the live model:

| | |
|---|---|
| Released automatically, no human | **6** — $23,265 |
| Refused outright | **4** — $112,650 withheld |
| Routed to a human | **15** — $149,034 |

So it pays 24% of invoices straight through and still puts 60% in front of a person.
Stated plainly, because the number matters and the honest reading is not the obvious one.

**The queue was never the expensive part. The investigation was.**

A clerk facing INV-1013 today opens a 22,562.80 invoice with eight lines, three of them
repeating SKUs at different prices, and has to work out by hand whether it adds up. It
does not — it is $50 short — but finding that takes twenty minutes of adding, and most
people do not do it, which is where a 30% error rate comes from. The same clerk now opens
a queue item that already says:

> `arithmetic.mismatch` — stated 22,562.80, computed 21,040.00 + 1,472.80 tax = 22,512.80
> `stock.insufficient` — WidgetA: 22 billed against 15 in stock (short 7)

with the vendor verified, the catalog matched, the duplicate ledger checked, and a written
rationale explaining what to do about it. The decision still needs a human. The *work* does
not. That is the five days.

### What it caught that a person reading carefully would probably miss

| | |
|---|---|
| Misstated totals, invisible by eye | **$1,410** across INV-1007, INV-1009, INV-1013 |
| Double payment prevented | **$3,000** — INV-1011 arrived as both a PDF and a text file; the second was caught against the ledger as a duplicate of one already paid |
| Fraud refused | **$109,900** — INV-1003 (unlisted vendor, discontinued part, wire-transfer pressure) and INV-1008 (unlisted vendor, two products that do not exist, priced $100 under the review threshold) |

Three of the twenty-five supplied invoices fail their own arithmetic, and not one of them
looks wrong. That is the case for recomputing every figure in Python rather than asking a
model whether it adds up.

### Where this goes next

The 60% review queue is one undifferentiated pile, and it should not be. Every item in it
already carries a machine-readable cause, and those causes belong to different desks — a
stock shortfall is a buyer's phone call, an 85% tax line is a finance question, a duplicate
is an AP check. Routing on the finding code turns one queue of fifteen into four short
lists of pre-investigated work, and that is a small change to a field that already exists.

Further out, and deliberately not built here: several of these refusals need no human at
all. An invoice from a vendor who is not on the approved list, for a part that does not
exist, is not a judgment call — the system already writes a rationale good enough to send.
Auto-rejecting on corroborated criteria and emailing the vendor the reasoning would close
the loop without a person touching it, and the audit trail to justify each one is already
being written. The reason it stops short of that here is that releasing outbound
communication on a model's say-so needs a confidence bar this prototype has not earned yet,
and the failure documented below is exactly why.

---

## Architecture

```
                     ┌──────────────────────────────┐
                     │      (arithmetic mismatch)   │
                     ▼                              │
    START ──▶ extract ──▶ reconcile ──┬─────────────┘
                                      ├──▶ validate ──▶ approve ◀──┐
                                      │                    │       │
                                      │                    ▼       │
                                      │                 critique ──┤ (revision
                                      └──▶ finalize ◀──────┘       │  recommended)
                                              │                    │
                                              ▼                    │
                                             END                   │
```

| Node | Kind | Job |
|---|---|---|
| `extract` | LLM | Document → typed `InvoiceData`. Transcription only. |
| `reconcile` | **Python** | Recompute every figure in `Decimal`, aggregate quantities per SKU. |
| `validate` | LLM + tools | Deterministic checks first; the agent rules on what's left. |
| `approve` | LLM | Weighs the findings and decides, under an enforced policy floor. |
| `critique` | LLM | Argues with the draft. May send it back. |
| `finalize` | **Python** | Pays if approved. Records every outcome to the ledger. |

Agents don't call each other. LangGraph nodes share one typed state object; each returns a
partial update that gets merged. Fields several nodes contribute to (`findings`, `errors`,
`audit_log`) carry `operator.add` reducers so they accumulate; single-author fields
(`invoice`, `decision`) overwrite. Getting that wrong fails silently — the validator's
findings simply vanish when the approver writes — so the reducers have their own tests
against a compiled graph.

A side effect worth naming: the final state is a complete audit trail. What was extracted,
what the arithmetic said, every finding with its evidence, the decision and its reasoning,
and a timestamped log of every step. For a business losing money to a 30% error rate, the
ability to say *why* a payment went out is most of the value, and here it falls out of the
architecture rather than being bolted on.

---

## Four design decisions

### 1. Python does the arithmetic, not the model

The extractor is forbidden from calculating anything. It transcribes what the document
*claims*; `reconciliation.py` computes what it *should say*; the two are compared. A model
that both transcribes and calculates is checking itself, and agrees with itself every time.

This pays off three ways:

**It repairs OCR damage.** INV-1012 prints a line total as `$3,500.O0` — a letter O. The
transcription of that column is garbage. 7 × 500 is not.

**It catches errors nothing else would.** INV-1013 bills eight lines that sum correctly to
21,040.00, with 7% tax of 1,472.80 — also correct. But the stated grand total is 22,562.80,
and 21,040.00 + 1,472.80 = **22,512.80**. The invoice is $50 wrong, and both components
check out individually. Asked "does this add up?", a language model says yes.

INV-1007 hides the same shape of error more cheaply: its three lines sum to 14,750.00 and
its 6% tax of 885.00 is right, but it states a total of 15,525.00 against a real
15,635.00 — **$110 short**. Three of the 25 supplied documents fail their own arithmetic
(INV-1007, INV-1009, INV-1013), and not one of them is obvious from reading it.

**It drives a self-correction loop.** A mismatch is ambiguous between *the extractor
misread* and *the vendor is wrong*, so it routes back to the extractor once. If the re-read
reconciles, it was a misreading. If it doesn't, the document really is inconsistent. The
retry prompt is explicit that figures must not be adjusted to force agreement — without
that, the loop degenerates into fudging numbers until they balance, destroying exactly the
signal it exists to produce.

Money is `Decimal` throughout, built from `str(value)` so a float arriving as
`1472.8000000000002` becomes exactly `1472.80`. Comparisons tolerate one cent, because
vendors round tax by their own conventions.

### 2. The validator adjudicates; it doesn't look things up

Whether 22 exceeds 15 is not a question for a language model. Stock levels, vendor list
membership, duplicate detection, price variance and date consistency all run in Python
first (`validation.py`).

What reaches the agent is what those checks *couldn't* settle. The case that justifies the
whole design is INV-1016's `WidgetC`:

```
lookup_item("WidgetC") → no exact match
                         candidates: WidgetA 0.857, WidgetB 0.857, GadgetX 0.571
```

It ties against two different SKUs at once. Any auto-resolving matcher books it against
whichever sorted first, inventing an order the company never placed — precisely the error
class this system exists to remove. The right answer is that a SKU differing in its
trailing character is a different product, and that's a judgment, not a threshold.

So lookups never silently resolve an inexact name. They report an exact hit, or they report
scored near misses and let the agent rule. Meanwhile `Widget A` and `WidgetA (rush order)`
normalize to exact hits and never trouble the model at all.

The agent has four tools (`lookup_item`, `lookup_vendor`, `check_stock`,
`find_prior_invoices`) and returns **judgments, not findings** — Python converts those into
`Finding` objects with stable codes, so the vocabulary can't drift with whatever the model
felt like calling something today.

### 3. Policy is enforcement, not prompt text

The approver is a model reasoning about an invoice, and its judgment is the useful part.
But some outcomes must not be reachable by reasoning at all. `policy.py` runs *after* the
agent decides:

- a critical finding makes approval **unreachable**, not discouraged
- an invoice at or above $10,000 can't auto-approve while warnings are open
- no total, or no invoice, means nothing to release

The floor **only ever downgrades**. An approval it forbids becomes `needs_review`, never a
rejection — "a human must look at this" is the honest outcome. A model that decides to
reject is never overruled into paying.

### 4. Two loops, both bounded, both load-bearing

The extraction retry (above) and the approval critique. The critique is two nodes with an
edge between them rather than one node reflecting internally: a model asked to reconsider in
the same breath tends to restate, and a real edge puts the reversal in the audit trail as
two events with the argument between them. On INV-1003:

```
approver.decided   needs_review: 3 critical findings, held for a human.
critic.critiqued   revision recommended. The indicators here are mutually
                   reinforcing, not independent question marks.
approver.revised   rejected: together they describe an invoice that should
                   never be paid.
critic.critiqued   round 2: draft upheld.
```

Both loops are bounded by counters in state. INV-1013 can never reconcile and a critic can
always find one more thing to say; without bounds, both run forever.

---

## A failure we found, and what it changed

On a live run against `grok-4.6`, the approver refused a $22,562.80 invoice like this:

> *"It is a duplicate resubmission of an invoice already processed on 2026-09-20 with an
> identical line set; releasing this copy would create a double-payment."*

The ledger was **empty**. There was no duplicate, no prior submission, and no
`duplicate.resubmission` finding anywhere in the model's input. The only real defect was a
$50 arithmetic mismatch. The rationale was fluent, specific, plausible, and about an event
that never happened — and a human reading it had no way to tell.

Prose can't be validated. A citation can. `ApprovalDecision` now carries
`driving_findings`: the approver must name the exact `code` of every finding it relied on,
and `policy.verify_citations` checks each one against the findings it was actually given.
Anything it names that wasn't supplied is a fabrication, and a decision resting on one
isn't a decision — it's redirected to a human with the unsupported code recorded:

```
approver.unsupported_citation   cited duplicate.resubmission, not among this invoice's
                                findings; decision redirected to a human
```

This applies to rejections as much as approvals. Refusing a good supplier over an invented
duplicate is its own kind of harm, and the fix would be worth little if it only guarded the
money going out.

The same run surfaced a second problem: twelve documents disappeared from the batch without
a trace, because a run that never produced an invoice wrote nothing to the ledger at all.
A failure being *less* visible than a success, in the one table whose job is traceability,
is backwards. `invoice_number` is now nullable and every attempt is recorded — what it was
reading, and what went wrong.

---

## What it catches

Findings carry a stable code, a severity, a human-readable message and structured evidence.

| Family | Codes | Example |
|---|---|---|
| `arithmetic` | `mismatch` | INV-1013 is $50 out, INV-1007 $110, INV-1009's subtotal $1,250 |
| `data` | `negative_quantity`, `zero_quantity`, `fractional_quantity`, `missing_unit_price`, `negative_unit_price`, `missing_item_name`, `duplicate_line`, `negative_total` | INV-1009 bills −5 units; INV-1019 bills the same line twice |
| `tax` | `negative_rate`, `implausible_rate` | INV-1017's −5% "tax"; INV-1018's 85% |
| `charges` | `disproportionate` | INV-1020's $4,800 freight on $1,000 of goods |
| `stock` | `insufficient`, `out_of_stock`, `discontinued`, `unknown_item` | INV-1013 bills 22 WidgetA against 15 |
| `catalog` | `unknown_item`, `name_corrected` | INV-1016's WidgetC isn't a product |
| `vendor` | `unapproved`, `blocked`, `unverified`, `missing`, `name_corrected` | Fraudster LLC is on no list |
| `duplicate` | `resubmission`, `revised`, `same_content` | INV-1004 submitted twice |
| `price` | `variance` | INV-1010's rush line at +20% over catalog |
| `date` | `due_unresolvable`, `due_before_issue`, `terms_mismatch`, `already_overdue` | INV-1003 is due "yesterday" |
| `currency` | `unexpected` | A USD invoice from a EUR vendor |
| `risk` | `signal` | Pressure language, a total parked under a threshold |

Findings are never authored by a model. The validator and approver return *judgments* —
an adjudication, a risk signal, a citation — and Python turns those into `Finding` objects
with codes from the fixed vocabulary above. A model cannot invent a finding type, and
`verify_citations` means it cannot invent an instance of one either.

Some things worth noting about how these are drawn:

**Aggregation, not per-line checks.** INV-1013 bills WidgetA on three separate lines
(15 + 5 + 2 = 22 against 15 in stock) and INV-1010 splits it across standard and rush lines.
Every individual line passes. Only the sum fails, so quantities are totalled per SKU — by
normalized name, so `WidgetA`, `Widget A` and `WidgetA (rush order)` land in one bucket.

**A stockout is not fraud.** `CoolantPro` has zero stock because it's backordered.
`FakeItem` has zero stock because it's delisted, and an inbound invoice for a delisted SKU
is a fraud signal. Collapsing both into "out of stock" loses the distinction that matters.

**Duplicates have two shapes.** The same invoice number resubmitted unchanged is a
`resubmission` (critical). The same number with different contents is a `revised` (warning —
plausibly legitimate). Different number, identical line items is `same_content`. All three
work because `finalize` writes **every** outcome to the ledger, rejections included, so a
refused vendor resubmitting under a fresh number still collides on the content hash.

### Invoices that add up perfectly and are still wrong

The five invoices in `data/invoices/` numbered 1017–1021 are ones I added, because the
supplied set tests extraction and stock but not plausibility. Each reconciles to the cent,
so the arithmetic layer has nothing to say about any of them:

| | What it hides |
|---|---|
| **INV-1017** | a −5% tax rate — a rebate dressed up as a statutory charge |
| **INV-1018** | an 85% tax rate, stated plainly and totalled correctly |
| **INV-1019** | WidgetB ×4 at $500 billed on two separate lines; stock isn't breached, so nothing else notices |
| **INV-1020** | $4,800 of freight on $1,000 of goods |
| **INV-1021** | no stated tax rate at all — but the amount works out to 40% of the goods |

The last one is why tax is checked on the *implied* rate as well as the stated one. An
invoice that prints only "Sales Tax: $1,100.00" never states anything implausible; the
rate has to be derived before there's anything to object to.

Shipping and tax get this attention for the same reason: unlike a line item, neither is
checked against anything. A billed part is validated against a catalog price and a stock
level. Freight is whatever the invoice says it is.

---

## The mock inventory database

`scripts/init_db.py` builds `inventory.db`. The schema is a strict superset of the brief's
— `item` and `stock` keep their names — with additions that support checks the base schema
can't express.

```
inventory       item, stock, unit_price, currency, category, status, notes, normalized
vendors         name, status (approved|unverified|blocked), payment_terms, currency,
                also_known_as, notes
invoice_ledger  invoice_number, revision, vendor_name, total, currency, line_hash,
                source_path, decision, reason, llm_mode, processed_at
```

### What `stock` is standing in for

Worth naming before anything else, because it is the one place this model is deliberately
wrong. Checking an **inbound** invoice against **our** stock level conflates two different
questions:

- *Did we order this much?* — a purchase-order match
- *Do we have it on hand?* — inventory

A vendor billing 20 GadgetX when the shelf holds 5 might be over-billing, or might simply
have shipped 20 that nobody has booked in yet. Only the first is an accounts-payable
problem, and stock cannot tell them apart.

A real system matches the invoice line against the PO line — ordered quantity, agreed
price, receipt confirmation — and `stock` is a one-column stand-in for that table because
the brief supplies it and a PO table would have been invention. Every stock finding should
be read as *"this quantity is unexpected, check the order"* rather than *"we are short."*
Swapping in a `purchase_orders` table is the single change that would most improve the
validator's precision, and nothing else in the design would have to move: the aggregation,
the per-SKU matching and the finding vocabulary all carry over unchanged.

The seed data is chosen to make validation a real job rather than a lookup:

- **CoolantPro** — a legitimate stockout, so a real supply problem is distinguishable from
  a bogus SKU
- **Fraudster LLC** and **NoProd Industries** are deliberately **absent** from `vendors`,
  so the two suspect invoices also fail vendor verification
- **QuickShip Distributors** is seeded with the correct spelling and marked `unverified`
  (a rebrand with changed banking details). INV-1012 spells it "Distributers", which
  surfaces as a 0.952 near miss for the agent to rule on — and correcting the spelling
  doesn't clear the underlying concern

---

## Testing

**254 tests, ~12s, no network.** The tests use the real invoice strings from `data/invoices/`, not invented ones — those are
what the pipeline actually has to survive. Coverage includes the arithmetic edge cases, the
fuzzy-matching boundaries, both retry loops terminating, the policy floor, graceful
degradation when a model call fails, and the state reducers against a compiled LangGraph.

Failure handling is tested as deliberately as the happy path. A backend error mid-run
doesn't kill the pipeline: the validator keeps its deterministic findings, the approver
holds the invoice for a human rather than defaulting either way, and `finalize` still
writes a ledger entry.

---

## Layout

```
main.py                      CLI
invoice_agents/
  graph.py                   the LangGraph pipeline
  state.py                   shared state and its reducers
  schemas.py                 every typed payload
  reconciliation.py          deterministic arithmetic
  validation.py              deterministic checks
  policy.py                  the approval floor
  payment.py                 mock banking API, ledger write
  tools.py                   LangChain tools over the database
  llm.py                     Grok / offline backend selection
  agents/
    extractor.py  validator.py  approver.py
    mock_*.py                offline stand-ins
  ingestion/loader.py        txt, json, csv, xml, pdf → text
  inventory/                 schema, seed, repository
  web/                       FastAPI console + static frontend
scripts/
  init_db.py                 build the database
  check_llm.py               verify the model backend
  record_fixtures.py         record real extractions for offline replay
  serve.py                   start the local console
```

Configuration is environment-driven (`.env`); `GROK_MODEL` defaults to `grok-4.6` against
`https://api.x.ai/v1`.

---

## Limitations

**No FX rates.** INV-1014 bills in EUR. The currency mismatch is flagged and cross-currency
price comparison is skipped rather than guessed at.

**No OCR.** `pdfplumber` reads embedded text. A scanned image would fail loudly with a
message saying so.

**Stock stands in for a purchase order.** Covered above — an inbound invoice is really
being matched against what was ordered, and `inventory.stock` is a one-column
approximation of that. It is the largest deliberate simplification in the design.

**The ledger is the only memory.** There's no vendor payment history, so "this vendor has
never billed above $500 before" isn't a signal the system can raise yet. That, and the
purchase-order table, are where the remaining fraud catches live.

**Single-invoice runs.** `--all` iterates sequentially. Invoices are independent, so this
parallelizes cleanly, but there was no reason to build that for 25 files.

**It is not fast.** One invoice costs five sequential model calls at minimum — extract,
the validator's tool loop and its verdict, approve, critique — rising to seven when the
arithmetic fails to reconcile or the critique forces a revision. Measured against
`grok-4.6`, a single invoice takes roughly 90 seconds:

```
 9,985ms  extractor.extracted
     0ms  reconciler.reconciled            ← Python, and it shows
42,338ms  validator.adjudicated            tool loop + structured verdict
14,228ms  approver.decided
25,164ms  critic.critiqued
```

That is a deliberate trade. Each call buys something the pipeline would otherwise have to
guess at, and against five days of manual handling, ninety seconds is not the bottleneck
anyone is complaining about. The obvious win if it ever mattered: when nothing needs
adjudication and the invoice carries no notes, the validator currently spends two calls
producing an empty verdict, and could skip the model entirely. `--all` would also
parallelize cleanly. Neither is built, because latency was never this system's problem —
the 30% error rate was.

Worth noting what costs nothing: the deterministic layer. Reconciliation recomputes every
figure, aggregates per SKU and runs the integrity checks in under a millisecond.
