# AstroLens — Scientific Dashboard

The frontend interface for the AstroVerse platform. Built with Next.js 16, Tailwind v4, and Recharts.

## Features
- **Detection Workspace:** Upload TESS light curve CSVs and run EvoMoE inference
- **Target Database:** Browse curated TESS targets from the MAST archive
- **Expert Routing Matrix:** Visualize how the MoE distributes decisions across CNN/Transformer/Physics experts
- **Report Generation:** Download transit analysis reports as Markdown

## Development

```bash
npm install
npm run dev
```

Requires the EvoNex API backend running on `http://localhost:8000`.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | EvoNex API endpoint |
