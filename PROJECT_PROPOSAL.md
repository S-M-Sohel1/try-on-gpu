# Fabric Try-On Preview — Project Proposal & Build Brief

> A staff-operated web tool for fabric shops. At the counter, staff photograph the
> customer and a fabric (or pick fabric from a library), choose a garment type, and an
> AI generates a realistic preview of the customer "wearing" a garment made from that
> fabric — helping the customer decide what to make and which fabric to buy.

**Document purpose:** This is a build brief intended to be fed to Claude Code as the
source-of-truth spec for the project. It states the product, the MVP scope, the
recommended technical approach, the data model, the AI integration contract, and a
phased roadmap. Recommendations are marked as such; open decisions are listed at the end.

---

## 1. Problem & Vision

Fabric shops (gaz/piece-cloth shops) sell raw fabric by the yard. Customers struggle to
imagine how a fabric will look as a finished garment (shirt, pant, panjabi, saree,
uniform, etc.). This causes hesitation, indecision, and lost sales.

**Vision:** Give shop staff a one-tap tool that turns "what would this look like as a
shirt on me?" into an instant visual preview, increasing customer confidence and sales.

**This is a visual preview tool, not a measurement or tailoring tool.** It makes no
claims about fit or sizing. (See Assumptions.)

---

## 2. Users

- **Primary user:** Shop counter staff (often non-technical). They operate the tool.
- **Beneficiary:** The walk-in customer, who views the preview and decides.
- **Admin:** The product operator (you), managing shops, fabric libraries, and usage
  centrally.

The interface must be usable by non-technical staff: large buttons, minimal typing,
single-screen flow.

---

## 3. Core User Flow (MVP)

1. **Start a session** at the counter.
2. **Capture the customer photo** using the tablet camera (front-facing, clear, well-lit).
3. **Choose the fabric**, via one of two paths:
   - **(a) From the fabric library** — search by shop-assigned code, scan a QR/barcode
     label, or visually browse a thumbnail grid.
   - **(b) Live photo** — photograph the fabric at the counter (fallback for fabric not
     in the library).
4. **Choose the garment type** — tap a preset (**Shirt** or **Pant** for MVP),
   optionally pick style details (sleeve length, collar, fit). Free-text exists but
   presets are the default path.
5. **Choose a template/style** — select from pre-defined silhouette templates for the
   chosen garment category.
6. **Tap Generate** (a deliberate action, not auto-fire — this controls cost).
7. **View the result.** Save / print / share. Optionally try another fabric or garment on
   the same customer photo.

---

## 4. Features

### 4.1 MVP (must-have)
- Camera capture of customer photo.
- Live fabric photo capture.
- Garment-type presets: **Shirt (upper-body)** and **Pant (lower-body)** only for v1.
- Template/silhouette selector per garment category.
- 2-stage AI pipeline: fabric → canonical garment image → try-on (see §6).
- Auto-mask generation for the person image — staff never draw a mask.
- Deliberate "Generate" button + clear loading state + progress indicator.
- Display result; save/download the output image.
- Graceful handling of generation failure and dropped connection (clear retry, never a
  blank screen).
- API credentials held server-side / GPU-worker-side only.
- Per-shop usage logging (metering) from day one, even if billing comes later.

### 4.2 Phase 2 (fabric library)
- Pre-uploaded, searchable fabric library per shop.
- Fabric identification: shop-assigned code + visual browse (primary), QR/barcode label
  scan (convenience). **Do not assume fabrics have manufacturer barcodes** — most raw
  bolts do not; the shop assigns its own codes/labels.
- Admin screen to add/edit/remove fabrics (photo, code, name, category, colour, material,
  price).

### 4.3 Phase 3 (extended garments & multi-shop)
- Add more garment categories: Panjabi / Kurti / Blouse → upper; Trouser / Skirt → lower;
  Dress / Jumpsuit → overall.
- Sharee and three-piece handled in a dedicated sub-pipeline (not mixed into the main
  flow).
- Multi-shop support; central backend serving all shops.
- Per-shop usage dashboard for the operator (catch outliers, monitor volume).
- **Swappable GPU worker** behind a common job-queue interface (see §6.4).

### 4.4 Phase 4 (business & scale)
- Subscription/billing for shops (see §9).
- Move GPU worker from Colab/ngrok to a managed GPU service (RunPod, Modal, etc.).
- Optional post-processing: upscale / cleanup pass on output images.

### 4.5 Nice-to-have (later, unprioritised)
- Side-by-side comparison of two fabrics on the same customer.
- WhatsApp share of the result.
- Customer-facing kiosk/self-service mode.
- Remote/at-home customer use.

---

## 5. Platform

- **Responsive web app, delivered as a PWA** ("Add to Home Screen"), running on an
  Android tablet at the counter.
- Rationale: single codebase, no app-store friction, instant updates, camera works in
  modern mobile browsers, and the device is a fixed counter tablet operated by staff —
  the ideal case for a web app over native.
- Lock orientation to the tablet's mounting (portrait or landscape) so layout is stable.

---

## 6. Recommended Technical Architecture

> All choices below are recommendations. Claude Code may substitute equivalents, but
> should preserve the **principles**: credentials server-side only, AI pipeline
> swappable, per-shop metering, fabric identification not dependent on manufacturer
> barcodes.

### 6.1 Core Architecture — 2-Stage Pipeline

**Do NOT feed raw fabric directly into the try-on model.** CatVTON (and similar VTON
models) are conditioned on a *garment image*, not a fabric swatch. Feeding a raw swatch
produces inconsistent outputs.

The correct pipeline is:

```
Fabric image  ──► [Stage A] Fabric → Canonical Garment  ──►  [Stage B] Garment → Try-On
                            (GPU Worker)                              (CatVTON / GPU Worker)
```

**Stage A — Fabric → Canonical Garment Image**
- Takes the raw fabric swatch photo.
- Applies it to a garment template/silhouette for the chosen category.
- Outputs a clean, front-view, flat garment image (e.g. a front-view shirt with that
  fabric's colour, pattern, and texture).

**Stage B — Garment Image → Person Try-On (CatVTON)**
- Inputs: person image, canonical garment image (from Stage A), auto-generated mask.
- Outputs: the final try-on result.

### 6.2 Category Mapping

| Business category                    | CatVTON category |
|--------------------------------------|-----------------|
| Shirt / Panjabi / Kurti / Blouse     | `upper`         |
| Pant / Trouser / Skirt               | `lower`         |
| Dress / Jumpsuit                     | `overall`       |

**MVP scope: `upper` (shirt) and `lower` (pant) only.**
Sharee and three-piece are excluded from v1 — they require a separate sub-pipeline and
will complicate the first release significantly.

### 6.3 Auto-Mask Generation (no user input required)

CatVTON requires a clothing-region mask for the person image. **Staff never draw a
mask.** The GPU worker generates it automatically based on garment category:

| Category | Mask target                               | Preserved region      |
|----------|-------------------------------------------|-----------------------|
| `upper`  | Torso + sleeve/arm clothing area          | Lower-body clothes    |
| `lower`  | Thigh/leg clothing area                   | Upper-body clothes    |
| `overall`| Full body clothing area                   | Face, hands, shoes    |

Recommended tooling: a human-parsing or segmentation model (e.g. SCHP, Grounded-SAM)
run on the GPU worker before CatVTON inference.

### 6.4 What Runs Where

```
┌─────────────────────────────┐
│  Counter Tablet (PWA)       │
│  - select category + style  │
│  - upload person image      │
│  - upload fabric image      │
│  - show status / result     │
└────────────┬────────────────┘
             │ HTTPS
             ▼
┌─────────────────────────────┐
│  Backend (Render / Node)    │
│  - auth / API / job mgmt    │
│  - store images             │
│  - basic validation         │
│  - basic resize/orientation │
│  - enqueue GPU job          │
│  - store result / job status│
└────────────┬────────────────┘
             │ job queue / HTTP
             ▼
┌─────────────────────────────┐
│  GPU Worker                 │
│  - Stage A: fabric →        │
│    canonical garment image  │
│  - Stage B: auto-mask gen   │
│    from person image        │
│  - Stage B: CatVTON infer   │
│  - optional: upscale/fix    │
└─────────────────────────────┘
```

### 6.5 Stack (recommended)

- **Frontend/Backend:** Next.js (App Router) + TypeScript. Bundles the PWA frontend and
  backend API routes in one deployable unit. Good fit for MVP.
- **Styling:** Tailwind CSS. Large tap targets, high contrast, minimal text input.
- **PWA:** service worker for installability + app shell; generation requires network.
- **Database + Storage + Auth:** Supabase (Postgres + object storage + auth).
  SQLite acceptable for a single-shop prototype.
- **Job queue:** Simple polling table in Supabase (MVP); upgrade to a proper queue
  (BullMQ, Cloud Tasks) if concurrency demands it.
- **GPU Worker:** FastAPI (Python) deployed initially on Google Colab + ngrok (see §6.6),
  then migrated to a managed GPU service (RunPod, Modal, etc.).

### 6.6 AI Provider Abstraction (important)

Define a single interface so the GPU worker can be swapped without rewriting the app:

```ts
interface TryOnProvider {
  generate(input: {
    personImage: Buffer | string;      // base64 or URL
    fabricImage: Buffer | string;      // base64 or URL
    garmentCategory: "upper" | "lower" | "overall";
    templateId?: string;               // silhouette template
  }): Promise<{ outputImage: string; modelUsed: string; }>;
}
```

Selected via a `TRYON_PROVIDER` environment variable.

---

## 7. AI Integration Details

### 7.1 MVP Engine — CatVTON

**CatVTON** (Concatenation-based Virtual Try-On) is the recommended open-source VTON
model for MVP. It is garment-image conditioned, supports upper/lower/overall mask modes,
and runs on a single consumer GPU (≥ 16 GB VRAM recommended).

> **Verify the exact CatVTON model weights and repo at build time.** The canonical
> source is the official CatVTON GitHub/HuggingFace page.

### 7.2 Full GPU Inference Pipeline (per generation request)

```
Input:  person_image, fabric_image, garment_category, template_id

Step 1 — Stage A: Fabric → Canonical Garment
  Load garment template silhouette for (garment_category, template_id)
  Generate a front-view garment image with the fabric's colour/pattern applied
  → canonical_garment_image

Step 2 — Stage B: Auto-Mask Generation
  Run segmentation / human-parsing model on person_image
  Select mask region based on garment_category (upper / lower / overall)
  → person_mask

Step 3 — Stage B: CatVTON Inference
  Inputs: person_image, canonical_garment_image, person_mask
  → final_output_image

Step 4 (optional) — Post-processing
  Upscale / cleanup / colour correction
  → final_output_image (polished)
```

### 7.3 Stage A — Fabric → Garment Image

For MVP, Stage A can be implemented as:
- **Option A (recommended):** Use a texture-transfer / inpainting approach with a
  template silhouette. Run a generative model (e.g. stable-diffusion-inpaint or
  FLUX-based) to apply the fabric texture to the garment template.
- **Option B (simpler fallback):** Simple texture-mapping / UV-wrap onto a flat garment
  template using image processing (OpenCV). Lower quality but zero GPU cost for Stage A.

Start with Option B during Colab development; upgrade to Option A once the pipeline is
validated end-to-end.

### 7.4 Quality Expectations (set with shop & customers)

Output is an **approximate preview**, not a guarantee of the stitched product. Solid
colours and simple patterns transfer best; complex prints/embroidery may be imperfect.
Clear, well-lit, front-facing customer photos give the best results.

---

## 8. Initial Development Workflow (Colab GPU)

During the initial development phase, the GPU worker runs on **Google Colab** with a
free/pro GPU runtime. This avoids provisioning a paid GPU server before the pipeline is
validated.

```
Local dev  ──► GitHub  ──► Colab (git clone/pull)  ──► FastAPI + ngrok  ──► Backend/App
   |                              |
  code                     GPU inference
  changes                  runs here
```

**Workflow steps:**
1. Develop GPU worker code locally (Python / FastAPI).
2. Push to GitHub.
3. In Colab: `git clone` / `git pull` the same repo.
4. Run the FastAPI app inside Colab (`uvicorn worker.main:app`).
5. Expose with **ngrok** (`ngrok http 8000`) — gives a public HTTPS URL.
6. Backend (Render) sends GPU jobs to that ngrok URL.
7. Test the full pipeline end-to-end.

> **Colab limitations:** session time-outs, no persistent storage, not suitable for
> production. Migrate to RunPod / Modal / similar when moving beyond development.

---

## 9. Cost Model & Scaling Path

CatVTON is an open model — the cost is GPU compute, not per-API-call fees.

| Phase                     | Infrastructure         | Approx. cost model              |
|---------------------------|------------------------|---------------------------------|
| Dev / testing             | Colab free/pro         | ~$0 – $12/mo                   |
| MVP (low volume)          | RunPod / Modal spot    | ~$0.20–0.50/GPU-hr on demand    |
| Growth (steady volume)    | RunPod reserved / Modal| Scale horizontally              |
| High volume               | Dedicated GPU node     | Evaluate vs. serverless         |

**Cost controls baked in:** deliberate Generate button; cache customer photo per session
so re-trying fabrics doesn't re-upload; soft per-shop usage caps; job queue prevents
GPU overload.

---

## 10. Business Model (parked — summary only)

- Per-shop **flat monthly subscription** with a **soft, unpublished fair-use limit**
  (Claude-style: limits exist but the exact number isn't published, giving pricing
  flexibility).
- **B2B nuance:** acknowledge to shop owners that fair-use exists; don't fully hide it.
  Enforce gently and off the counter (warn the owner out-of-band, never hard-block a
  generation mid-customer). Reserve hard limits for genuine abuse.
- Use a **rolling window** rather than a hard monthly cliff.
- **Internal metering must be precise even if the shop sees no number** — this is how you
  spot a client whose usage costs more than they pay and move them to a higher tier.
- A hard customer-facing credit gate is **not** recommended for the counter app.

---

## 11. Data Model (recommended)

```
shops
  id, name, location, status, subscription_tier, created_at

staff
  id, shop_id (FK), name, role, auth_ref, created_at

fabrics
  id, shop_id (FK), code (shop-assigned), name, category, colour,
  material, price_per_unit, image_url, qr_value (nullable), created_at

garment_templates           // silhouette templates per category
  id, category (upper|lower|overall), name, preview_image_url,
  style_options (json), created_at

jobs                        // GPU job queue
  id, shop_id (FK), staff_id (FK), status (queued|processing|done|failed),
  person_image_url, fabric_image_url, garment_category, template_id,
  canonical_garment_url (nullable), output_image_url (nullable),
  error_message (nullable), created_at, updated_at

generations                 // metering + history (written when job completes)
  id, shop_id (FK), staff_id (FK), job_id (FK),
  fabric_id (FK, nullable), garment_category, template_id,
  output_image_url, model_used, gpu_seconds (nullable),
  cost_estimate, created_at
```

Per-shop usage is derived from `generations`.

---

## 12. API Contract (recommended)

- `POST /api/jobs` — body: person image, fabric reference (id OR image), garment
  category, template id → creates a job, returns `job_id` + `status`. Logs a `jobs` row.
- `GET /api/jobs/:id` — poll for job status + result URL.
- `GET /api/fabrics?shop_id=` — list/search fabric library.
- `POST /api/fabrics` / `PATCH` / `DELETE` — manage fabrics (admin).
- `GET /api/usage?shop_id=` — per-shop usage summary (operator dashboard).
- Auth endpoints (shop/staff login).

**GPU Worker API (internal, called by backend):**
- `POST /infer` — body: person_image_url, fabric_image_url, garment_category,
  template_id → runs full pipeline → returns output_image (base64 or URL).

All generation traffic goes tablet → backend → GPU worker. The client never calls the
GPU worker directly.

---

## 13. Non-Functional Requirements

- **Performance:** generation takes several seconds (GPU pipeline); always show a loading
  state with progress steps ("Generating garment…", "Applying to photo…").
- **Reliability:** friendly errors + retry on failure or lost connectivity.
- **Security:** GPU worker not publicly accessible (only callable from backend); AI
  credentials server-side only; per-shop auth; least-privilege storage access.
- **Privacy:** define a customer-photo retention policy (default recommendation: delete
  after session unless the customer opts in to saving). Make it explicit and configurable.
- **i18n:** Bangla + English interface (toggle). (Confirm — see Assumptions.)
- **Accessibility/usability:** large tap targets, high contrast, minimal typing.

---

## 14. Suggested Build Order (for Claude Code)

1. **Phase 1 — GPU Worker skeleton (Colab):**
   - FastAPI app with `/infer` endpoint.
   - Stage A: simple texture-map fabric onto garment template (OpenCV, no GPU needed yet).
   - Stage B: auto-mask generation (human-parsing model).
   - Stage B: CatVTON inference.
   - Expose via ngrok; test with static images.

2. **Phase 2 — Backend + PWA MVP (single-shop):**
   - PWA shell, customer photo capture, live fabric photo.
   - Garment preset UI: Shirt (upper) and Pant (lower) only.
   - Template/style selector.
   - `POST /api/jobs` + `GET /api/jobs/:id` polling.
   - Result display + save/download; error handling.
   - Basic `jobs` + `generations` logging. Hardcode one shop.

3. **Phase 3 — Fabric library:**
   - `fabrics` table, library search + visual browse + QR/barcode scan.
   - Admin CRUD screen; fabric-from-library path in generate flow.

4. **Phase 4 — Multi-shop + ops:**
   - shops/staff/auth, per-shop metering, operator usage dashboard.
   - Migrate GPU worker from Colab to managed GPU service.

5. **Phase 5 — Extended garments & business/scale:**
   - Add Panjabi/Kurti/Trouser/Skirt/Dress categories.
   - Subscription + soft limits.
   - Stage A upgrade to generative texture transfer.
   - Sharee / three-piece in separate sub-pipeline.

---

## 15. Assumptions (confirm or correct)

- This is a **staff-operated** tool; customers do not use it directly (no self-service in
  MVP).
- It is a **visual preview only** — no sizing/measurement/fit claims.
- Interface languages: **Bangla + English**.
- Fabrics generally **do not** carry manufacturer barcodes; the shop assigns its own
  codes/labels.
- Output actions for MVP: **save/download**; print and WhatsApp share are later.
- Customer photo retention: **deleted after session by default** (privacy-first); revisit
  if saving is desired.
- **MVP garment scope: Shirt (upper) and Pant (lower) only.** Sharee and three-piece are
  explicitly out of v1.

---

## 16. Open Decisions (need your input before/while building)

1. Output actions: which of save / print / WhatsApp / side-by-side compare are in MVP?
2. Customer-photo retention policy — confirm the default above.
3. Does the shop owner get any dashboard, or is all admin centralised with the operator?
4. How many style sub-options per garment (and which) for the presets?
5. Confirm target tablet OS/browser (assumed Android + Chrome).
6. Stage A method: start with simple OpenCV texture-map, or go straight to a generative
   approach (adds GPU cost and complexity to Stage A)?
7. Auto-mask model choice: SCHP, Grounded-SAM, or another human-parsing model?
8. CatVTON model weights: which checkpoint / version to use?
9. Colab tier: free (T4) or Colab Pro (A100) for initial development?
