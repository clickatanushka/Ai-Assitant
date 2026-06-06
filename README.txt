AUDIT ASSISTANT — SETUP GUIDE
==============================

WHAT YOU GET
------------
• Q&A tab: Ask questions in English → answers shown side-by-side in English + German
  with exact page number and paragraph citations from your 81 PDFs
• Dashboard tab: Enter titles + Actual/Target values for up to 8 charts → clean visual graphs

FOLDER STRUCTURE
----------------
audit_system/
├── server.py       ← the backend (run this)
├── index.html      ← the frontend (auto-served)
├── pdfs/           ← PUT ALL YOUR 81 PDFs HERE
├── index.json      ← auto-created when you index (do not edit)
└── api_key.txt     ← auto-created when you save your API key

STEP 1 — Install Python requirements
-------------------------------------
Open a terminal in the audit_system folder and run:

    pip install anthropic pdfplumber

That's all you need.

STEP 2 — Add your PDFs
-----------------------
Copy all 81 German PDF files into the   pdfs/   folder.

STEP 3 — Start the server
--------------------------
In terminal:

    python server.py

You will see:
    ✅ Server running at http://localhost:8000

STEP 4 — Open the app
----------------------
Open your browser and go to:

    http://localhost:8000

STEP 5 — Add API key
---------------------
1. Click the ⚙ Settings button (top right)
2. Paste your Anthropic API key  (get one at https://console.anthropic.com)
3. Click Save Key

STEP 6 — Index your PDFs
-------------------------
Click the  ↺ Re-index PDFs  button in the top bar.
This runs ONCE and extracts text from all 81 PDFs (~2-5 minutes).
The index is saved — you don't need to redo it unless you add new PDFs.

STEP 7 — Ask questions!
------------------------
Type any question in English and press Ask (or Enter).
You will see:
• Left column: Answer in German (for your auditor)
• Right column: Answer in English (for you)
• Below: Each source citation with the original German paragraph + English translation,
  showing which PDF file and which page number it came from.

DASHBOARD TAB
-------------
• Fill in the graph title + Actual + Target for each of up to 8 metrics
• Click "Update Charts" to see the bar charts with % achievement
• Your data is saved automatically in the browser

SHARING THIS SYSTEM
--------------------
To give this to someone else:
1. Send them the entire  audit_system/  folder
2. They need Python installed (python.org)
3. They run:  pip install anthropic pdfplumber
4. They add their own pdfs/ folder
5. They run:  python server.py
6. They open:  http://localhost:8000

TROUBLESHOOTING
---------------
"No documents indexed"  → Make sure PDFs are in the pdfs/ folder, then click Re-index
"No API key"            → Click Settings and paste your key
"Error reading PDF"     → Some scanned PDFs (images only) can't be read without OCR;
                          text-based PDFs work best
Server not starting     → Make sure port 8000 is free, or edit PORT = 8000 in server.py
