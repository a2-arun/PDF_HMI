# grade_pdfs/

Place the prescribed textbook PDFs here, named to match `config.py`:

```
grade_pdfs/
├── Grade6.pdf
├── Grade7.pdf
├── Grade8.pdf
└── Grade9.pdf
```

These files are intentionally **not** committed to the repository (see
`.gitignore`) — they're typically copyrighted, large, and specific to
whichever curriculum you're deploying against. After adding them, build the
knowledge base with:

```
python ingest.py
```
