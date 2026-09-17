# Local general documents

Save HTML, UTF-8 TXT, or PDF files under `data/raw/general/`, then run
`python scripts/parse_general_docs.py` from the installed project environment.
The parser reads local files only. It writes `general_documents.json` and
`general_documents_manifest.json` in `data/processed/`.

Use category folder names or descriptive filenames: `research_degree_overview`,
`entry_requirements`, `how_to_apply`, `scholarships_and_fees`, `faq`, or `other`.
`research_degrees_overview` and `research_degree_faq` are also recognized.
For example, `faq/questions.pdf` or `entry_requirements_2026-09-15.html`.
Unknown names become `other`; category folders take precedence over filenames.

HTML source URLs come from saved canonical or Open Graph metadata. PDF/TXT URLs
and unknown save dates remain null. File modification time is not assumed to be
the original save date. IDs are stable hashes of the relative filename and page
number. Separate files remain separate records even when their contents match.

Each nonempty PDF page becomes one record with its original 1-based page number.
No OCR is performed. Blank or image-only pages are reported in `empty_pdf_pages`;
files with no extracted text are reported in `empty_files`. Failed files are
reported individually while other files continue. Unsupported extensions are ignored.
Manifest category counts refer to document records; file-type counts refer to
source files, including empty or failed ones (`htm` is counted separately).

HTML cleanup removes common navigation, cookie banners, and UTAS sidebar menus.
It retains headings, paragraphs, and collapsed FAQ text; layout-specific cleanup
is heuristic. No chunking or requests to source URLs are performed.
