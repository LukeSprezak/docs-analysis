# Frontend — Docs Analysis client

React SPA for the Docs Analysis tool. Migrated from Create React App to **Vite + React 19 + TypeScript 6**.

## Stack

| Area | Tool | Version |
|---|---|---|
| Build/dev server | Vite | 8.x (`vite.config.ts`) |
| React plugin | @vitejs/plugin-react | 6.x |
| UI library | React + ReactDOM | 19.x |
| Language | TypeScript | 6.x (`moduleResolution: "bundler"`) |
| Styling | Tailwind CSS | 3.4 (`darkMode: 'class'`) |
| Markdown | react-markdown 10 + remark-gfm 4 | pure ESM |
| Toasts | react-hot-toast | 2.x |

Package manager is **npm** (`package-lock.json` is the lockfile — there is no yarn.lock).

## Commands

Run from `client/`:

```bash
npm install        # install deps
npm run dev        # dev server (Vite) on http://localhost:3000
npm run build      # tsc --noEmit && vite build  → output in dist/
npm run preview    # serve the production build locally
npm run typecheck  # tsc --noEmit only
```

Docker (from repo root): `docker compose up --build -d frontend`. The image build runs `npm ci && npm run build`; the prod stage serves `dist/` via Caddy (`docker/caddy/Caddyfile`, which also handles SPA fallback and `/api` proxying).

## Project layout (`src/`)

Organized by role so the app scales without a flat `src/`:

```
src/
  index.tsx, App.tsx, index.css, vite-env.d.ts   # entry + app shell
  core/         api.ts                            # data layer + shared types
  contexts/     AuthContext, LanguageContext      # React context providers
  components/   Markdown, icons, confirmDelete     # shared, reusable UI
  features/     ChatTab, QATab, SummarizerTab,     # feature views (one per tab)
                FAQTab, LoginScreen
```

- `index.tsx` — entry; `createRoot` + `<StrictMode>`. Mounted from root `index.html` via `<script type="module" src="/src/index.tsx">`.
- `App.tsx` — shell: tab switching (chat / qa / summarizer / faq), dark-mode toggle.
- `core/api.ts` — all backend calls. JWT in `localStorage`, `authedFetch` adds `Authorization: Bearer`; a 401 clears the token and fires the unauthorized handler.
- `contexts/` — `AuthContext` / `LanguageContext`. Translations are **fetched from the backend** (`api.getTranslations(lang)`), not bundled.
- `components/` — shared UI: `Markdown` (styled react-markdown wrapper), `icons` (inline SVGs, sized via `className`), `confirmDelete` (shared delete-confirmation toast).
- `features/` — one file per tab/view.

**Where new code goes:** reusable UI → `components/`; a new tab/screen → `features/`; backend calls and shared types → `core/`; cross-cutting React state → `contexts/`. Imports are relative (`../core/api`); there is no path alias configured.

## Conventions

- **Naming: full descriptive names, no abbreviations** — variables, functions, classes (e.g. `processingInterval`, not `procInt`). Applies to new code too.
- **Env vars**: Vite, not CRA. Use `import.meta.env.VITE_*` (e.g. `import.meta.env.VITE_API_URL`). `process.env.REACT_APP_*` no longer works. Only client-safe values — anything in `VITE_*` ships to the browser.
- **PostCSS/Tailwind configs are `.cjs`** (`postcss.config.cjs`, `tailwind.config.cjs`) because `package.json` has `"type": "module"`. Keep them CommonJS or convert fully to ESM — don't mix.
- **Timers**: type with `ReturnType<typeof setTimeout>` / `setInterval`, not `NodeJS.Timeout` (this is a browser app; `@types/node` is not installed).

## React 19 idioms (prefer these)

- **`ref` is a regular prop** — function components take `ref` directly; do not wrap new components in `forwardRef`.
- **Actions** — for form/async mutations use `useActionState` and `useOptimistic`; `useTransition` actions can be async and expose `isPending`.
- **`use(...)`** — read promises/context conditionally instead of threading through `useEffect` where it fits.
- `<Context>` can be rendered directly as a provider (no `.Provider` needed).
- Document metadata (`<title>`, `<meta>`) can be rendered inside components and hoists to `<head>`.

## TypeScript 6 notes

- `moduleResolution: "bundler"` — import without extensions; Vite resolves.
- Inferred type predicates (since 5.5): a function returning a boolean narrowing check infers `x is T` automatically — lean on it instead of hand-written guards where possible.

## Gotchas

- **Port 3000 + CORS**: `api.ts` calls the backend at an **absolute** URL (`VITE_API_URL` or `http://localhost:8001/api/v1`), bypassing the Vite dev proxy. The backend's CORS allowlist expects origin `http://localhost:3000`, so the dev server **must** run on 3000. If 3000 is taken (e.g. the dockerized `frontend` container), Vite silently falls back to 3001 and backend calls — including translations — fail CORS. Free port 3000 (`docker compose stop frontend`) rather than accepting the fallback.
- The Vite proxy (`/api` → `:8001` in `vite.config.ts`) is currently a safety net only; real traffic uses the absolute URL above. To make the port irrelevant, switch `API_BASE_URL` to the relative `/api/v1` and route everything through the proxy — a deliberate change, not done yet.
- react-markdown is pure ESM (v10); its component map drops the old `inline` arg on `code`. The current `Markdown.tsx` map is compatible.