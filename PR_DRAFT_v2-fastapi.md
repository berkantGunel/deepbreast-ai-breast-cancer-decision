# Draft: Merge `v2-fastapi` -> `main`

## Title

Merge: v2.0.0 — FastAPI backend + React (Vite) frontend redesign

## Summary

This PR replaces the previous Streamlit-based prototype (v1) with a production-oriented FastAPI backend and a modern React + Vite frontend (v2). Key changes:

- Rebuilt frontend into `deepbreastai/` (Vite, React, TypeScript, Tailwind). Pages: Home, Predict, Analysis, Metrics, About.
- FastAPI backend in `src/api/` with endpoints: `/api/predict`, `/api/gradcam`, `/api/metrics`, `/api/training-history`.
- Version bumped to `2.0.0` (`version/__version__.py`) and tag `v2.0.0` created.
- Removed test/temp files and cleaned repo surface.

## Files/areas touched (high level)

- `deepbreastai/` — new/updated frontend app
- `src/api/` — FastAPI endpoints and utilities
- `models/` — model files and eval/train JSON used by endpoints
- `version/__version__.py` — version bumped to `2.0.0`

## Migration / upgrade notes

- Backend: ensure virtual environment has required packages (see `requirements.txt`). Start uvicorn: `python -m uvicorn src.api.main:app --reload --port 8000`.
- Frontend: install and build in `deepbreastai/`: `npm install` then `npm run dev` (or `npm run build` for production).
- If deploying, verify CORS settings and the `API_BASE_URL` used by the frontend.

## Checklist (pre-merge)

- [ ] Run backend locally and verify `/api/health` and `/api/metrics` respond.
- [ ] Start frontend dev server and verify Home, Predict, Analysis, Metrics pages load and call backend endpoints.
- [ ] Add screenshots/GIFs showing the new UI (place in PR or attach assets).
- [ ] Confirm no sensitive files or tokens are present in the branch.
- [ ] If `main` is production, create a backup tag of current `main` before merge: `git tag -a backup-main-before-v2 -m "backup main before v2 merge"` and `git push origin backup-main-before-v2`.

## Suggested PR body (copy-paste to GitHub PR)

Title: `feat(release): v2.0.0 — FastAPI backend + React (Vite) frontend redesign`

Body:

```
This PR introduces the v2.0.0 release: a full migration from the Streamlit prototype to
a FastAPI backend and a React + Vite frontend. Highlights:

- FastAPI endpoints for inference, Grad-CAM, and metrics.
- React frontend with Streamlit-like layout: responsive sidebar, prediction page, Grad-CAM analysis, and metrics visualizations.
- Version bumped to 2.0.0 and tag `v2.0.0` created.

Please review the changes and run the local checks listed in the PR checklist. If `main` is protected,
I recommend creating a Draft PR and running CI before merging.

Release notes (summary):
- New: FastAPI backend
- New: React + Vite frontend with Tailwind
- Breaking: `main`'s Streamlit app is replaced by v2 — keep `main` backup if you want to preserve v1.

```

## Notes for reviewers

- Backend API shapes were normalized to match frontend expectations (flattened metrics JSON, training history converted to row-wise structure).
- If CI is present, ensure necessary secrets (if any) are configured for running tests or builds.

---

Place screenshots in `docs/screenshots/` (create folder) or attach via GitHub PR UI.
