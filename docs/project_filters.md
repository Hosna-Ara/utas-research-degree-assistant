# Explicit project filters

`filter_projects(projects, **filters)` accepts validated `ProjectDocument` records.
All supplied filters must match. Supported keys are `degree_type`, `student_type`,
`location`, `funding_status`, `status`, `research_category`, `supervisor`, and
`project_id`. Matching is case-insensitive with whitespace cleaned; list values
use exact membership, and semicolon-separated locations are treated separately.
Supervisor and project ID matching are exact, not substring searches.

Aliases include MRes/Masters by Research, international student, scholarship
(funded), open (Applications open), and ICT. `AI/ICT` in probe C means the
explicit Information and Communication Technology category. It does not classify
projects as AI-related using their descriptions. Funding uses the existing
retrieval-only label; it is not a guarantee of scholarship availability.

```bash
python scripts/filter_projects.py --degree PhD --student-type International --location Hobart --funding funded --status open
python scripts/filter_projects.py --probe
```

The CLI prints counts and the first ten matches in source order. No ranking or
automatic query-to-filter extraction is performed. Source records are not edited.
