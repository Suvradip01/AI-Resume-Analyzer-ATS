# InSightATS Frontend — Complete Interview Guide

> **Companion doc:** [BACKEND-INTERVIEW-GUIDE.md](./BACKEND-INTERVIEW-GUIDE.md) covers the FastAPI ML pipeline. This document covers the **React + Vite** app in `frontend/`, deployed on **Vercel** (`vercel.json` SPA rewrites).

---

## Table of Contents

1. [What the frontend does](#1-what-the-frontend-does)
2. [High-level architecture](#2-high-level-architecture)
3. [Tech stack](#3-tech-stack)
4. [Repository layout (every file)](#4-repository-layout-every-file)
5. [Bootstrap and routing](#5-bootstrap-and-routing)
6. [Authentication (Clerk)](#6-authentication-clerk)
7. [Environment variables](#7-environment-variables)
8. [Landing page](#8-landing-page)
9. [Candidate dashboard](#9-candidate-dashboard)
10. [Recruiter flows](#10-recruiter-flows)
11. [Charts and data visualization](#11-charts-and-data-visualization)
12. [UI component library](#12-ui-component-library)
13. [Styling and design system](#13-styling-and-design-system)
14. [Build, deploy, and tooling](#14-build-deploy-and-tooling)
15. [Backend integration summary](#15-backend-integration-summary)
16. [Interview talking points](#16-interview-talking-points)
17. [Function index by file](#17-function-index-by-file)

---

## 1. What the frontend does

InSightATS is a **single-page application (SPA)** with two product surfaces:

| Surface | Route(s) | User | Core action |
|---------|----------|------|-------------|
| **Marketing site** | `/` | Public | Explain product, contact form, CTAs |
| **Candidate analyzer** | `/dashboard` | Clerk-signed-in job seeker | Upload resume + paste JD → scores + feedback |
| **Recruiter shortlist** | `/recruiter`, `/recruiter/dashboard` | Clerk recruiter (`unsafeMetadata.company`) | Upload many resumes + JD file → ranked list |

Auth is entirely **Clerk** (`@clerk/clerk-react`). The backend analyze endpoint for candidates is **unauthenticated**; recruiter batch calls send a **Clerk JWT**.

---

## 2. High-level architecture

```mermaid
flowchart TB
    subgraph entry [Entry]
        HTML[index.html]
        MAIN[main.jsx]
        CLERK[ClerkProvider]
        ROUTER[BrowserRouter]
    end

    subgraph routes [App.jsx Routes]
        LAND[LandingPage]
        SIGNIN[/sign-in]
        SIGNUP[/sign-up]
        DASH[/dashboard]
        RECLOGIN[/recruiter]
        RECDASH[/recruiter/dashboard]
    end

    subgraph external [External]
        API[FastAPI Backend]
        CLERKAPI[Clerk Cloud]
        EMAIL[EmailJS]
    end

    HTML --> MAIN --> CLERK --> ROUTER --> routes
    DASH -->|POST multipart| API
    RECDASH -->|Bearer JWT| API
    SIGNIN --> CLERKAPI
    RECLOGIN --> CLERKAPI
    LAND --> EMAIL
```

**Data flow (candidate analyze):**

```text
User selects file + types JD
  → handleAnalyze() builds FormData
  → fetch(VITE_API_BASE + /api/v1/resume/analyze)
  → JSON: score, *_score, feedback[], missing_skills[]
  → ResultPieChart + ResultRadarChart + categorized cards
```

---

## 3. Tech stack

| Package | Version (approx) | Role |
|---------|------------------|------|
| **React** | 19 | UI library |
| **Vite** | 7 | Dev server, HMR, production bundle |
| **React Router** | 7 | Client-side routes |
| **Clerk** | 5 | Auth (sign-in, sign-up, sessions, JWT) |
| **Tailwind CSS** | 4 | Utility styling via `@tailwindcss/vite` |
| **Framer Motion** | 12 | Page/chart animations |
| **@visx/*** | 3 | Custom SVG pie + radar charts |
| **Lenis** | 1 | Smooth scroll on landing |
| **Lucide React** | icons | UI icons |
| **EmailJS** | 4 | Contact form without backend |
| **class-variance-authority** + **clsx** + **tailwind-merge** | — | shadcn-style `cn()` + Badge/Alert variants |

**Listed but unused in source (interview honesty):**

- `recharts` — charts use **Visx**, not Recharts.
- `@number-flow/react` — no imports in `src/`.

---

## 4. Repository layout (every file)

```
frontend/
├── index.html                 # Shell: #root, loads /src/main.jsx
├── package.json               # Scripts + dependencies
├── package-lock.json
├── vite.config.js             # React plugin, Tailwind, @ alias
├── jsconfig.json              # Path alias @/* → src/*
├── components.json            # shadcn metadata (style: new-york)
├── eslint.config.js           # ESLint 9 flat config
├── vercel.json                # SPA rewrite: all routes → index.html
├── README.md                  # npm install + dev + API note
├── public/
│   └── assets/
│       ├── dashboard-preview.png
│       └── feedback.png
│       # Note: our-latest-creation.jsx references ./assets/score.png — file missing in repo
└── src/
    ├── main.jsx               # React root, ClerkProvider, BrowserRouter
    ├── App.jsx                # Route table + guards
    ├── global.css             # Tailwind + theme tokens + chart CSS vars
    ├── lib/
    │   └── utils.js           # cn() helper
    ├── pages/
    │   ├── Dashboard.jsx          # Candidate analyze UI
    │   ├── sign-in.jsx            # Clerk SignIn
    │   ├── sign-up.jsx            # Clerk SignUp
    │   ├── recruiter-login.jsx    # Custom recruiter auth UI
    │   └── recruiter-dashboard.jsx
    ├── sections/              # Landing-only sections
    │   ├── hero-section.jsx
    │   ├── our-latest-creation.jsx
    │   ├── about-our-apps.jsx
    │   └── get-in-touch.jsx
    └── components/
        ├── navbar.jsx
        ├── flip-words.jsx
        ├── tilt-image.jsx
        ├── lenis-scroll.jsx
        ├── section-title.jsx
        ├── ResultPieChart.jsx
        ├── ResultRadarChart.jsx
        └── ui/                  # shadcn-style primitives
            ├── card.jsx
            ├── badge.jsx
            ├── alert.jsx
            └── progress.jsx
```

---

## 5. Bootstrap and routing

### `index.html`

- Sets title **InSightATS**, viewport meta.
- Single mount point `<motion.div id="root">` — Vite injects `main.jsx` as ES module.

**Why minimal HTML:** Vite convention; all UI is React.

---

### `src/main.jsx`

| Piece | Purpose |
|-------|---------|
| `import './global.css'` | Tailwind + fonts load before paint |
| `VITE_CLERK_PUBLISHABLE_KEY` | **Required** — throws if missing (fail fast) |
| `ClerkProvider` | Wraps app; `afterSignOutUrl="/"` |
| `BrowserRouter` | Enables `Routes` / `Link` / `useNavigate` |
| `createRoot(...).render(...)` | React 19 root API |

**Why Clerk at root:** Session context must wrap any route using `useUser`, `SignedIn`, `getToken`.

---

### `src/App.jsx`

**Component `RequireRecruiter`**

- Uses `useUser()` → `isLoaded`, `isSignedIn`.
- Loading: full-screen spinner.
- Not signed in: `<Navigate to="/recruiter" />`.
- Else: render children.

**Note:** Does **not** check `unsafeMetadata.company` — only that *some* Clerk session exists. Company is enforced implicitly because recruiters sign up via `recruiter-login.jsx` with metadata.

**Component `LandingPage`**

Composes: `LenisScroll`, `Navbar`, and sections inside `<main>` with responsive horizontal padding.

**Route table:**

| Path | Element | Guard |
|------|---------|-------|
| `/` | `LandingPage` | None |
| `/sign-in/*` | `SignInPage` | None |
| `/sign-up/*` | `SignUpPage` | None |
| `/recruiter` | `RecruiterLoginPage` | None |
| `/recruiter/dashboard` | `RecruiterDashboardPage` | `RequireRecruiter` |
| `/dashboard` | `Dashboard` | `<SignedIn>` else `<RedirectToSignIn />` |

**Why `/*` on Clerk paths:** Clerk’s hosted components use sub-routes (e.g. factor verification).

---

## 6. Authentication (Clerk)

### Two user personas

Detection is **client-side** via Clerk `user.unsafeMetadata`:

```javascript
const isRecruiter = isSignedIn && user?.unsafeMetadata?.company;
const isCandidate = isSignedIn && !user?.unsafeMetadata?.company;
```

| Persona | How they register | Metadata | Dashboard |
|---------|-------------------|----------|-----------|
| **Candidate** | `/sign-up` — Clerk default | No `company` | `/dashboard` |
| **Recruiter** | `/recruiter` custom form | `unsafeMetadata: { company }` | `/recruiter/dashboard` |

**Why `unsafeMetadata`:** Writable from client on sign-up; good for demo. Production interviews: mention you'd use **publicMetadata** set server-side or Clerk organizations for stricter roles.

---

### `pages/sign-in.jsx` & `pages/sign-up.jsx`

Thin wrappers around Clerk components:

```jsx
<SignIn path="/sign-in" routing="path" signUpUrl="/sign-up" />
<SignUp path="/sign-up" routing="path" signInUrl="/sign-in" />
```

**Why separate pages:** Candidates get hosted Clerk UI; recruiters use custom branded flow.

---

### `pages/recruiter-login.jsx` — Custom auth (important for interviews)

**State machine `mode`:**

- `login` | `register` | `verify_signup` | `forgot_password` | `reset_password`

**Key functions:**

| Function | Behavior |
|----------|----------|
| `submit(e)` | Validates fields; branches on `mode` |
| Password rules (register/reset) | Regex: ≥8 chars, one digit, one special char |
| `signUp.create({ emailAddress, password, unsafeMetadata: { company } })` | Stores company name |
| `signUp.prepareEmailAddressVerification` + `attemptEmailAddressVerification` | Email OTP flow |
| `signIn.create` | Standard login |
| Forgot password | `strategy: "reset_password_email_code"` |

**UI:** Split panel — branding left (desktop), form right; `framer-motion` slides panels when toggling login/register.

**On success:** `setSignUpActive` / `setSignInActive` → `nav("/recruiter/dashboard")`.

**Does NOT call** backend `/api/v1/recruiter/register` — auth is **100% Clerk** in production UI.

---

### `components/navbar.jsx` — Auth-aware navigation

**State:**

- `isMenuOpen` — mobile drawer
- `isDropdownOpen`, `dashDropOpen`, `signDropOpen` — desktop dropdowns

**Desktop logic (three branches):**

1. **Signed out** (`SignedOut`): “Dashboard” dropdown → Candidate vs Recruiter dashboard links; “Sign In” dropdown → `/sign-in` vs `/recruiter`.
2. **Recruiter signed in:** Link to `/recruiter/dashboard`; custom avatar chip + logout (calls `signOut()`).
3. **Candidate signed in:** Link to `/dashboard`; Clerk `<UserButton />`.

**Mobile:** Full-screen overlay with same links.

**Why dual dashboard links when signed out:** Lets users deep-link before Clerk session exists; protected routes still redirect.

---

## 7. Environment variables

All Vite env vars must be prefixed with **`VITE_`** to expose to client.

| Variable | Used in | Purpose |
|----------|---------|---------|
| `VITE_CLERK_PUBLISHABLE_KEY` | `main.jsx` | Clerk SDK (required) |
| `VITE_API_BASE` | `Dashboard.jsx`, `recruiter-dashboard.jsx` | Backend origin (no trailing slash) |
| `VITE_EMAILJS_SERVICE_ID` | `get-in-touch.jsx` | Contact form |
| `VITE_EMAILJS_TEMPLATE_ID` | `get-in-touch.jsx` | Email template |
| `VITE_EMAILJS_PUBLIC_KEY` | `get-in-touch.jsx` | EmailJS public key |

**Local defaults:**

- Dashboard: `VITE_API_BASE ?? "http://127.0.0.1:8000"`.
- Recruiter dashboard: **no fallback** — `VITE_API_BASE` must be set on Vercel.

**Security note for interviews:** Never put Clerk secret key or backend secrets in `VITE_*` — they ship in the browser bundle.

---

## 8. Landing page

### `components/lenis-scroll.jsx`

| Function | Purpose |
|----------|---------|
| `useEffect` | Instantiates `Lenis({ duration: 1.2, smoothWheel: true, anchors: true })` |
| `requestAnimationFrame` loop | `lenis.raf(time)` |
| cleanup | `lenis.destroy()` |

**Why Lenis:** Premium smooth scroll for marketing; disabled on dashboard pages (not mounted there).

---

### `sections/hero-section.jsx`

**Marketing content:**

- Animated SVG radial gradient background.
- Badge: “AI Powered Resume Analysis”.
- Headline with `<FlipWords words={["Dream job", "Next Step", "True Path"]} />`.
- Subcopy: “Transparent 4D scoring…”
- CTAs: `<a href="/dashboard">` (candidate), `<a href="/recruiter">` (recruiter).
- `<TiltedImage />` — dashboard screenshot.

**Animations:** `whileInView` + spring transitions on scroll into view.

---

### `components/flip-words.jsx`

| Export | Props | Logic |
|--------|-------|-------|
| `FlipWords` | `words`, `duration=3000`, `className` | Cycles words with blur/spring letter animation |

**Implementation details:**

- Ghost invisible `longestWord` reserves layout width (prevents layout shift).
- `AnimatePresence` + `onExitComplete` → triggers next word after `duration`.
- Per-letter staggered `MotionSpan` for cinematic effect.

---

### `components/tilt-image.jsx`

| Function | Purpose |
|----------|---------|
| `handleMouse` | Maps cursor offset → `rotateX` / `rotateY` via `useSpring` |
| `handleMouseLeave` | Resets rotation |
| Renders | `dashboard-preview.png` with 3D perspective |

**Why:** Hero visual polish; demonstrates Framer Motion + motion values.

---

### `sections/our-latest-creation.jsx` — `id="creations"`

**State:**

- `activeIndex` — carousel (auto-advance every 3s).
- `isHovered` — pauses auto-advance.
- `className` — enables width transition after first `whileInView` animation.

**`sectionData`:** Three feature cards with images (parsing, feedback, ATS). **Bug/ gap:** `score.png` path used but file not in `public/assets/`.

---

### `sections/about-our-apps.jsx` — `id="about"`

**`FeatureCard` (memo):** Icon + title + description; light cards on dark page (marquee).

**`featuresData`:** Six items mapping to backend concepts (parsing, NER, semantic match, project analysis, explainability, 4D scoring).

**Marquee CSS:** Duplicated array `[...featuresData, ...featuresData]` for infinite scroll; row 2 uses `marquee-reverse`; hover pauses animation.

---

### `sections/get-in-touch.jsx` — `id="contact"`

| Function | Purpose |
|----------|---------|
| `handleSubmit` | Validates; `emailjs.sendForm(SERVICE_ID, TEMPLATE_ID, form, { publicKey })` |
| Placeholder check | If `SERVICE_ID === "your_service_id"` → friendly dev error |

**Form fields:** `from_name`, `email`, `message` (EmailJS template must match).

**Status UI:** `idle | loading | success | error` with motion alerts.

**Why EmailJS:** No backend email endpoint; keeps contact on static hosting.

---

### `components/section-title.jsx`

**Props:** `title`, `description`.

**Effect:** Splits title — last word gets gradient underline animation via `backgroundSize` on `whileInView`.

**Used by:** creations, about, contact sections.

---

## 9. Candidate dashboard

### `pages/Dashboard.jsx` — Core product page

**Imports of note:** Clerk `UserButton`, `useUser`; charts; shadcn `Card`, `Badge`, `Progress`, `Alert`; Framer `motion`, `AnimatePresence`.

**Constants:**

```javascript
const API_URL = `${import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000"}/api/v1/resume/analyze`;
```

#### State

| State | Type | Purpose |
|-------|------|---------|
| `file` | `File \| null` | Selected resume |
| `jobDescription` | string | Pasted JD text |
| `loading` | boolean | Disables button, shows spinner |
| `result` | object \| null | API JSON |
| `error` | string | User-facing error |

#### `handleFileChange(e)`

- Accepts MIME: PDF, DOCX, TXT only.
- Else sets error message.

#### `handleAnalyze()` — Main API integration

1. Validates `file` + `jobDescription`.
2. `FormData`: `resume_file`, `job_description` as JSON string `{ title: "Target Role", description }`.
3. `fetch(API_URL, { method: "POST", body: formData })` — **no Authorization header**.
4. On error: `response.json()` → `err.detail`.
5. On success: `setResult(data)`; scroll to `#results-section`.

**Why FormData not JSON:** Backend expects multipart (`UploadFile` + `Form`).

#### `cleanFeedbackText(text)` — Presentation layer

Backend returns markdown-ish strings with emojis and labels. This function:

1. Strips unicode artifacts (`\uFFFD`, etc.).
2. Rewrites `**Name (Basic Level)**:` patterns for project cards.
3. Strips labels like `Skills:`, `Experience Relevance:`, `Tip for Project:`.
4. Converts `**bold**` → `<strong class="text-white">` for `dangerouslySetInnerHTML`.

**Why client-side:** Backend feedback is stable contract; UI wants cleaner cards without changing ML strings.

**Interview caveat:** `dangerouslySetInnerHTML` — only safe because you control backend text; XSS if API compromised.

#### `parseFeedback(feedbackItem)` — Card taxonomy

Maps substring patterns → `{ type, status, title, subtitle, badge, icon }`:

| Pattern | type | UI |
|---------|------|-----|
| `✅ **Skills**` | skills | success |
| `❌ **Critical Missing Skills**` | skills | warning |
| Basic/Medium Level | project | info |
| Tip for Project | project (tip) | info |
| Advanced Level | project | success |
| Experience Relevance | experience | warning |
| Overall | overall | Final verdict styling |

**Must stay in sync** with `app/services/feedback/builder.py` on backend.

#### `getCategorizedFeedback()`

Buckets `result.feedback[]` into `{ skills, projects, experience, other }` for grid layout.

#### UI structure

1. **Fixed header** — logo, fake “System Operational” dots, `UserButton`.
2. **Input grid** — file upload card + JD textarea.
3. **Run Analysis** button — Framer error banner.
4. **Results** (AnimatePresence):
   - `ResultPieChart score={result.score}`
   - `ResultRadarChart` with four axes from `skill_score`, `experience_score`, `project_score`, `structure_score`
   - Categorized `Card`s with `Badge`, `Progress`, missing skill chips

---

## 10. Recruiter flows

### `pages/recruiter-dashboard.jsx`

**Auth:**

```javascript
const API_BATCH = `${import.meta.env.VITE_API_BASE}/api/v1/recruiter/batch-analyze`;
```

- `useEffect`: redirect to `/recruiter` if `!isSignedIn`.
- `getToken()` from Clerk → `Authorization: Bearer ${token}`.
- Header `X-Company-Name`: `user?.unsafeMetadata?.company || "your company"`.

#### State

| State | Purpose |
|-------|---------|
| `files` | Array of resume `File`s |
| `jdFile` | Single JD file |
| `loading`, `error`, `data` | Request lifecycle |

#### `analyze()`

1. Builds `FormData`: multiple `resumes`, one `job_description_file`.
2. POST with auth headers.
3. Sets `data` to `BatchAnalyzeResponse` shape.

#### `getRankIcon(rank)` / `getRankStyle(rank)`

Visual treatment for ranks 1–3 (gold/silver/bronze gradients).

#### `cleanFeedbackText` / `getFeedbackIcon`

Same ideas as Dashboard; shows first 3 feedback lines per candidate.

#### Results UI

For each `data.results[]`:

- `r.rank`, `r.filename`, `r.score` (donut SVG with `strokeDasharray` animation).
- `r.analysis.fit_result.verdict`, `missing_skills`.
- Five `Progress` bars: AI Fit (`fit_score * 100`), Skills, Exp, Proj, Struct.

**Why duplicate cleanFeedback:** Shared util candidate — could extract to `lib/feedback.js` (refactor talking point).

---

## 11. Charts and data visualization

### `ResultPieChart.jsx` — Headline score donut

**Library:** `@visx/shape` `Pie`, `@visx/group`, `@visx/scale` `scaleOrdinal`.

**Props:** `{ score }` (0–100).

**Data:** Two slices — `score` (violet) and `100 - score` (slate).

**Interactions:**

- Hover on score slice → scale 1.05, glow gradient, tooltip with percentage.
- Center label: animated `{Math.round(score)}` + “Total Score”.

**Why Visx not Recharts:** Full control over SVG + Framer `motion.path` per arc; lighter than configuring Recharts for one donut.

---

### `ResultRadarChart.jsx` — “XAI Match Breakdown”

**Props:** `data = [{ subject, A, fullMark: 100 }, ...]` — four metrics from Dashboard.

**Not a classic radar:** Renders **four overlapping “petal” polygons** — one per axis, each peaked on its own axis and symmetric wings on neighbors (`primaryValue * 0.65`).

**`getAdjustedValue(val)`:** Minimum visible radius 16 when value > 0 (so tiny scores still show).

**Colors:**

- Skills → blue `#3b82f6`
- Experience → green `#10b981`
- Projects → amber `#f59e0b`
- Structure → violet `#8b5cf6`

**UX:** Hover highlights one series + legend row; tooltips on vertices.

**Interview line:** “We visualize the same four axes the backend computes in `derive_scores` — not raw model logits.”

---

## 12. UI component library

Built in **shadcn “new-york”** style (`components.json`) but committed as plain JSX (no Radix in repo — simplified primitives).

### `lib/utils.js`

```javascript
export function cn(...inputs) {
  return twMerge(clsx(inputs))
}
```

Merges Tailwind classes without conflicts — used everywhere.

---

### `components/ui/card.jsx`

Exports: `Card`, `CardHeader`, `CardTitle`, `CardDescription`, `CardContent`, `CardFooter` — `forwardRef` + `cn()`.

**Used in:** Dashboard feedback cards.

---

### `components/ui/badge.jsx`

**`badgeVariants`** (CVA): `default`, `secondary`, `destructive`, `outline`, `success`, `warning`, `info`.

Dashboard uses `success`, `destructive`, `warning`, `info`, `outline` for skill chips.

---

### `components/ui/alert.jsx`

`Alert`, `AlertTitle`, `AlertDescription` — variants include `success`, `warning`, `info`.

Imported in Dashboard but primary errors use custom red divs; Alert available for extension.

---

### `components/ui/progress.jsx`

**Props:** `value` (0–100), optional `indicatorClassName` (recruiter dashboard sets per-metric colors).

**Implementation:** Inner bar `translateX(-${100 - value}%)` — CSS-only progress.

---

## 13. Styling and design system

### `src/global.css`

| Section | Content |
|---------|---------|
| Google Font | Poppins all weights |
| `@import "tailwindcss"` | Tailwind v4 entry |
| `@theme` | `--font-poppins`, colors: `background` #000, `foreground` #fff, `primary` white, `secondary` neutral-900 |
| `body` | `bg-background text-foreground font-poppins antialiased` |
| `:root` chart vars | oklch chart palette, grid, label colors for Visx/SVG |

**Design language:** Dark mode first; white primary buttons; glassmorphism (`backdrop-blur`, `border-white/10`); grid background on dashboards.

---

## 14. Build, deploy, and tooling

### `vite.config.js`

- Plugins: `@vitejs/plugin-react`, `@tailwindcss/vite`.
- Alias: `@` → `./src` (matches `jsconfig.json`).

### `vercel.json`

```json
{ "rewrites": [{ "source": "/(.*)", "destination": "/index.html" }] }
```

**Why:** Client-side routes (`/dashboard`, `/recruiter/dashboard`) must serve `index.html` on refresh.

### Scripts (`package.json`)

| Script | Command |
|--------|---------|
| `dev` | `vite` → usually `http://127.0.0.1:5173` |
| `build` | Production bundle to `dist/` |
| `preview` | Serve `dist` locally |
| `lint` | ESLint |

### `eslint.config.js`

- Flat config; React Hooks + React Refresh for Vite.
- `no-unused-vars` ignores vars matching `^[A-Z_]` (components).

---

## 15. Backend integration summary

| Page | Method | URL | Auth | Body |
|------|--------|-----|------|------|
| Dashboard | POST | `/api/v1/resume/analyze` | None | `resume_file`, `job_description` (JSON string) |
| Recruiter | POST | `/api/v1/recruiter/batch-analyze` | Bearer Clerk JWT | `resumes[]`, `job_description_file` |

**Response fields used (candidate):**

```javascript
result.score
result.skill_score, experience_score, project_score, structure_score
result.feedback[]      // strings
result.missing_skills[]
```

**Response fields used (recruiter):**

```javascript
data.total
data.results[].rank, filename, score
data.results[].analysis.{ fit_result, skill_score, experience_score, project_score, structure_score, feedback, missing_skills }
```

See [BACKEND-INTERVIEW-GUIDE.md](./BACKEND-INTERVIEW-GUIDE.md) for how those numbers are computed.

---

## 16. Interview talking points

### Elevator pitch (30 seconds)

> “The frontend is a React 19 Vite SPA with Clerk for dual personas — candidates and recruiters — a marketing landing page with smooth scroll and EmailJS contact, and two dashboards that talk to our FastAPI backend. Candidates get pie and custom Visx radar charts plus parsed feedback cards; recruiters upload batches and see ranked results with per-axis progress bars and Clerk JWT auth.”

### Common questions

1. **How do you separate recruiter vs candidate?**  
   Clerk `unsafeMetadata.company` set at recruiter registration; navbar and routes branch on that.

2. **Why is candidate analyze public but dashboard gated?**  
   Product choice: UI requires Clerk to access `/dashboard`; API itself is open (mention API rate-limiting in production).

3. **How do you handle CORS?**  
   Backend `ALLOWED_ORIGINS` must include Vercel URL; frontend only uses `fetch` to `VITE_API_BASE`.

4. **Why `dangerouslySetInnerHTML` on feedback?**  
   Backend sends `**markdown**` and emojis; `cleanFeedbackText` sanitizes partially — discuss moving to a markdown renderer + DOMPurify.

5. **Chart library choice?**  
   Visx for bespoke SVG + Framer; Recharts in package.json is unused legacy.

6. **SPA routing on Vercel?**  
   `vercel.json` rewrites all paths to `index.html`.

7. **What would you improve?**  
   Extract shared `cleanFeedbackText`; add `VITE_API_BASE` fallback on recruiter page; fix missing `score.png`; verify Clerk JWT on backend; add React Query for analyze loading states.

### User journey diagrams

**Candidate:**

```text
/ → Sign In (/sign-in) → /dashboard → upload + JD → Run Analysis → charts + cards
```

**Recruiter:**

```text
/ → /recruiter → register (company metadata) → verify email → /recruiter/dashboard
→ upload resumes + JD file → Generate Shortlist → ranked cards
```

---

## 17. Function index by file

### `Dashboard.jsx`

- `handleFileChange`, `handleAnalyze`
- `cleanFeedbackText`, `parseFeedback`, `getCategorizedFeedback`
- default export `Dashboard`

### `recruiter-dashboard.jsx`

- `getRankIcon`, `getRankStyle`, `cleanFeedbackText`, `getFeedbackIcon`
- `logout`, `analyze`
- default export `RecruiterDashboardPage`

### `recruiter-login.jsx`

- `submit`, `toggleMode`
- default export `RecruiterLoginPage`

### `get-in-touch.jsx`

- `handleSubmit`

### `flip-words.jsx`

- `FlipWords`, internal `startAnimation`

### `tilt-image.jsx`

- `handleMouse`, `handleMouseLeave`
- default export `TiltedImage`

### `navbar.jsx`

- default export `Navbar` (no named exports)

### `ResultPieChart.jsx`

- default `ResultPieChart({ score })`

### `ResultRadarChart.jsx`

- `getColorForMetric`, `getAdjustedValue`, `generateSeriesPolygon`, `getPoint`
- default `ResultRadarChart({ data })`

### `App.jsx`

- `RequireRecruiter`, `LandingPage`, default `App`

### `lib/utils.js`

- `cn`

---

*You now have paired guides: **BACKEND-INTERVIEW-GUIDE.md** + **FRONTEND-INTERVIEW-GUIDE.md**. Practice walking a interviewer from landing page → Clerk auth → API call → chart binding using both docs.*
