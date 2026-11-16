# Contributing to Inkray

Contributions to **Inkray** are welcome - whether through **issues** (bug reports, UI feedback, feature ideas) or **pull requests** (bug fixes, new tools, tests, and documentation improvements).

## Ways to Contribute

- **Open an Issue:** Report bugs with reproduction steps, environment details (OS, GPU/CPU, Python/Node versions), screenshots, or error logs, or start a discussion before making large architectural changes.
- **Submit a Pull Request:** Fork the repository, create a focused branch from `main`, implement and test your changes, and open a PR with a concise summary of what changed and why.

## Development Workflow

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/<your-username>/inkray.git
   cd inkray
   ```
2. Set up the backend environment (`conda` or `venv`, see [backend/README.md](backend/README.md)) and frontend dependencies (`cd frontend && npm install`).
3. Make focused, clean changes matching existing conventions:
   - **Backend (Python):** Keep modules cleanly separated (geometry, appearance, inference, cache, API), prefer explicit type annotations, and avoid unnecessary complexity or fallback layers.
   - **Frontend (TypeScript / React / WebGL2):** Follow strict TypeScript and ESLint rules and keep code formatted with Prettier.

## Verification Before Opening a PR

Run the backend unit tests:

```bash
cd backend
pytest -v
```

Run all frontend checks and production build:

```bash
cd frontend
npm run typecheck
npm run lint
npm run format:check
npm test
npm run build
```

If you modify ML inference or compositing behavior, run the standalone backend scripts when hardware/weights are available:

```bash
cd backend
python scripts/smoke_moge.py
python scripts/smoke_marigold.py --with-moge
```
