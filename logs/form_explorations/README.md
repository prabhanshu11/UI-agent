# Google Forms Exploration - Methodology & Scripts

**Project:** UI-Agent Form Exploration
**Date:** January 11, 2026
**Target:** Anthropic Connectors Directory - Server Review Form
**Purpose:** Document Google Forms structure for future agent training

---

## Overview

This directory contains the results of an automated exploration of a Google Form used by Anthropic for reviewing MCP (Model Context Protocol) servers for their Connectors Directory. The exploration was conducted using Selenium WebDriver with Firefox and geckodriver.

### Goals

1. **Capture all form content** - Questions, options, and structure
2. **Document the exploration process** - Methods, challenges, and solutions
3. **Create training data** - For future UI automation agents
4. **Generate human-readable documentation** - For reference and analysis

---

## Directory Structure

```
form_explorations/
├── README.md                        # This file - methodology and usage
├── FORM_DOCUMENTATION.md            # Complete form content documentation
├── page_source.html                 # Raw HTML source from the form
├── extracted_content.txt            # Raw text extraction (177KB, includes JS/CSS)
├── page_1.png                       # Introduction page screenshot
├── page_2.png                       # Start of checklist
├── page_3.png - page_33.png         # Duplicate pages (pagination loop issue)
├── full_form_top.png                # Top of form (improved script)
└── scroll_001.png - scroll_003.png  # Scrolling screenshots (improved script)
```

---

## Exploration Scripts

### 1. google_form_explorer.py

**Location:** `~/Programs/UI-agent/src/bench/google_form_explorer.py`

**Purpose:** Initial attempt at automated form exploration with page-by-page navigation

**Features:**
- Opens Google Forms URL in Firefox
- Attempts to detect and click "Next" buttons
- Extracts questions and checkbox options
- Captures screenshots of each page
- Saves data to JSON and text files

**Issues Encountered:**
- Got stuck in pagination loop (clicked same element 31+ times)
- Question text extraction incomplete (showed "..." instead of full text)
- CSS selector fragility with Google Forms' obfuscated class names

**Results:**
- 33 screenshots captured (31 duplicates)
- 3 unique pages captured before loop
- Did not complete full form exploration

**Usage:**
```bash
cd ~/Programs/UI-agent
uv run src/bench/google_form_explorer.py
```

---

### 2. manual_form_extraction.py

**Location:** `~/Programs/UI-agent/src/bench/manual_form_extraction.py`

**Purpose:** Improved script using scrolling instead of pagination

**Features:**
- Scrolls through form incrementally
- Captures screenshot at each scroll position
- Extracts HTML page source
- Saves all visible text content
- Keeps browser open for manual inspection

**Improvements Over v1:**
- Avoids pagination loop issue
- Captures entire form on single page
- More reliable HTML source extraction
- Allows manual verification

**Results:**
- Complete HTML source captured (144KB)
- 3 scroll position screenshots
- Full text extraction (177KB including metadata)
- Successfully extracted all form content

**Usage:**
```bash
cd ~/Programs/UI-agent
uv run src/bench/manual_form_extraction.py
# Press Ctrl+C to close browser when done inspecting
```

---

## Extraction Methodology

### Phase 1: Automated Selenium Exploration

1. **Initialize WebDriver**
   ```python
   options = Options()
   driver = webdriver.Firefox(options=options)
   driver.set_window_size(1920, 1080)
   ```

2. **Navigate to Form**
   - Load form URL
   - Wait for elements to render (3-5 second delays)
   - Click initial "Next" button to proceed past introduction

3. **Content Extraction Attempts**
   - Method A: Detect question elements by CSS selectors
   - Method B: Scroll and capture visible text
   - Method C: Extract HTML source and parse

4. **Screenshot Capture**
   ```python
   driver.save_screenshot("page_X.png")
   ```

### Phase 2: HTML Source Analysis

After automated extraction, manual analysis of `page_source.html` proved most effective:

```bash
# Extract checkbox labels (most accurate method)
grep -oP 'aria-label="[^"]*"[^>]*role="checkbox"' page_source.html

# Extract section headers
grep -oP '"M7eMe">[^<]+<' page_source.html

# Find question containers
grep -oP 'freebirdFormviewerComponentsQuestionBaseTitle' page_source.html
```

### Phase 3: Manual Documentation

1. Read screenshots to verify content
2. Parse HTML for accurate text extraction
3. Organize findings into structured markdown (FORM_DOCUMENTATION.md)
4. Document lessons learned and recommendations

---

## Key Findings

### Google Forms Structure

**Technical Details:**
- Dynamic JavaScript rendering (React-based)
- ARIA attributes for accessibility (reliable selectors)
- Obfuscated CSS class names (unreliable selectors)
- Multi-page capable, but this form uses single page with sections

**Element Types Detected:**
- Checkboxes: `role="checkbox"` with `aria-label` for text
- Section headers: `class="M7eMe"` (may vary)
- Required markers: Red asterisk (*) in section titles

### Challenges

1. **Pagination Loop**
   - Problem: Script clicked "Next" button repeatedly on same page
   - Cause: Button selector matched element that didn't change page state
   - Solution: Use scrolling instead of pagination detection

2. **Text Extraction Noise**
   - Problem: Extracted content included 150KB+ of JavaScript/CSS
   - Cause: Selected all visible page elements
   - Solution: Parse HTML with targeted selectors

3. **Dynamic Class Names**
   - Problem: CSS classes like `aDTYNe snByac` are obfuscated
   - Cause: Google Forms uses minified/generated class names
   - Solution: Use ARIA labels and semantic HTML attributes

### Solutions

| Challenge | Solution | Effectiveness |
|-----------|----------|---------------|
| Pagination loops | Scroll + screenshot | ✅ High |
| Text extraction noise | HTML source + regex | ✅ High |
| Element detection | ARIA labels | ✅ High |
| CSS class unreliability | Semantic attributes | ✅ High |
| Form state validation | Screenshot hash comparison | 🔄 Not implemented |

---

## Form Content Summary

**Form Title:** Connectors Directory - Server Review Form

**Purpose:** Pre-submission checklist for MCP servers seeking directory inclusion

**Structure:**
- 1 introduction page
- 1 main checklist page
- 4 required sections
- 17 total checkbox items

**Sections:**
1. Policy Compliance (5 items)
2. Technical Requirements (5 items)
3. Documentation Requirements (4 items)
4. Testing Requirements (3 items)

**See FORM_DOCUMENTATION.md for complete content.**

---

## Tools & Dependencies

### System Requirements

```bash
# Firefox browser (pre-installed on Omarchy Linux)
firefox --version
# Output: Mozilla Firefox 146.0

# Geckodriver (Selenium WebDriver for Firefox)
geckodriver --version
# Output: geckodriver 0.36.0

# Python environment
python --version
# Output: Python 3.13
```

### Python Dependencies

Installed via `uv`:

```toml
[dependencies]
selenium = "^4.39.0"
pillow = "*"          # Image processing (indirect dependency)
```

Additional Selenium dependencies:
- attrs
- outcome
- pysocks
- sniffio
- sortedcontainers
- trio
- trio-websocket
- typing-extensions
- urllib3
- websocket-client
- wsproto

### Installation

```bash
cd ~/Programs/UI-agent

# Add Selenium to project
uv add selenium

# Geckodriver (if not installed)
cd /tmp
curl -sL https://github.com/mozilla/geckodriver/releases/download/v0.36.0/geckodriver-v0.36.0-linux64.tar.gz -o geckodriver.tar.gz
tar -xzf geckodriver.tar.gz
mv geckodriver ~/.local/bin/
chmod +x ~/.local/bin/geckodriver
```

---

## Reproducing the Exploration

### Step 1: Run Initial Exploration Script

```bash
cd ~/Programs/UI-agent
uv run src/bench/google_form_explorer.py
```

**Expected Output:**
- Console output showing page navigation
- Screenshots in `logs/form_explorations/page_*.png`
- (Will get stuck in loop around page 3)

### Step 2: Run Improved Extraction Script

```bash
uv run src/bench/manual_form_extraction.py
```

**Expected Output:**
- Console output showing scroll positions
- Screenshots in `logs/form_explorations/scroll_*.png`
- HTML source saved to `page_source.html`
- Extracted text in `extracted_content.txt`
- Browser remains open for inspection

### Step 3: Extract Form Content from HTML

```bash
cd ~/Programs/UI-agent/logs/form_explorations

# Extract checkbox labels
grep -oP 'aria-label="[^"]*"[^>]*role="checkbox"' page_source.html

# Extract section headers
grep -oP '"M7eMe">[^<]+' page_source.html | sed 's/"M7eMe">//'

# View page source in browser (optional)
firefox page_source.html
```

### Step 4: Review Documentation

```bash
# Read the complete form documentation
cat FORM_DOCUMENTATION.md

# View screenshots
feh full_form_top.png page_1.png page_2.png page_3.png
```

---

## Lessons for Future Agent Training

### What Agents Should Learn

1. **Resilience to UI Changes**
   - Use semantic HTML and ARIA attributes over CSS classes
   - Validate page state changes after navigation actions
   - Implement timeout and loop detection

2. **Multi-modal Approaches**
   - Combine visual (screenshots) + structural (HTML) + interactive (Selenium)
   - Don't rely on single extraction method
   - Vision models (GPT-4V, Claude with vision) can supplement HTML parsing

3. **Google Forms-Specific Patterns**
   - Checkboxes use `role="checkbox"` and `aria-label`
   - Section headers often have class `M7eMe` (but verify with ARIA)
   - Required sections marked with asterisk and `aria-required`
   - Forms may be single-page or multi-page (detect dynamically)

4. **Debugging Strategies**
   - Save HTML source for offline analysis
   - Capture screenshots at each step
   - Log all actions and element detections
   - Implement hash comparison to detect stuck states

### Recommended Agent Architecture

```python
class GoogleFormExplorer:
    def explore(self):
        # 1. Take initial screenshot
        self.capture_screenshot("initial")

        # 2. Try page navigation
        while self.has_next_button():
            page_hash = self.get_page_hash()
            self.click_next()
            time.sleep(2)

            # Detect loop
            if self.get_page_hash() == page_hash:
                print("Pagination not advancing, switching to scroll mode")
                break

            self.capture_screenshot(f"page_{self.page_num}")
            self.page_num += 1

        # 3. Fall back to scrolling
        self.scroll_and_capture()

        # 4. Extract HTML source
        self.save_html_source()

        # 5. Parse with multiple methods
        self.parse_with_selectors()
        self.parse_with_vision_model()
        self.parse_with_llm(self.html_source)

        # 6. Merge and validate results
        self.merge_extractions()
```

---

## Future Improvements

### For Exploration Scripts

- [ ] Implement page hash comparison for loop detection
- [ ] Add vision model integration (GPT-4V or Claude with Computer Use)
- [ ] Create reusable `GoogleFormsParser` class
- [ ] Support forms with file uploads, dropdowns, and text inputs
- [ ] Generate JSON schema from extracted form structure
- [ ] Add form submission simulation (with test data)

### For Documentation

- [ ] Auto-generate documentation from extraction results
- [ ] Create visual form flow diagrams
- [ ] Add form completion examples
- [ ] Document error messages and validation rules
- [ ] Include form submission response examples

### For Agent Training

- [ ] Create labeled dataset of form elements and their types
- [ ] Build training examples for common failure modes
- [ ] Develop evaluation metrics for form extraction accuracy
- [ ] Simulate various form layouts for robustness testing

---

## Integration with UI-Agent Repository

This exploration is part of the broader UI-Agent project:

```
UI-agent/
├── src/
│   ├── vision/          # LLM-based UI understanding
│   ├── android/         # Android device control
│   ├── core/            # Agent control loop (PPE cycle)
│   ├── perception/      # Input monitoring
│   └── bench/           # Benchmarks and explorations
│       ├── google_form_explorer.py       # This exploration
│       ├── manual_form_extraction.py     # This exploration
│       ├── server.py                     # Benchmark server
│       └── index.html                    # Benchmark UI
├── logs/
│   └── form_explorations/  # This directory
└── README.md
```

**Related Components:**
- `src/vision/llm_vision.py` - Could be used for screenshot analysis
- `src/android/screen.py` - Similar element detection patterns
- `src/core/perception.py` - Could integrate form exploration as task

---

## References

### Google Forms

- [Google Forms Help](https://support.google.com/docs/answer/6281888)
- [Google Forms API](https://developers.google.com/forms/api/guides)

### Selenium WebDriver

- [Selenium Python Docs](https://selenium-python.readthedocs.io/)
- [Firefox WebDriver (geckodriver)](https://github.com/mozilla/geckodriver)
- [ARIA Attribute Reference](https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA)

### Related Projects

- [Anthropic Computer Use API](https://www.anthropic.com/news/computer-use)
- [Claude Agent SDK](https://github.com/anthropics/anthropic-sdk-python)
- [MCP (Model Context Protocol)](https://modelcontextprotocol.io/)

---

## Contact & Attribution

**Explored by:** Claude Code (Anthropic's CLI tool)
**Human Operator:** prabhanshu11 (mail.prabhanshu@gmail.com)
**Repository:** https://github.com/prabhanshu11/UI-agent
**Date:** January 11, 2026

This exploration was conducted for educational and agent training purposes. The target form belongs to Anthropic and is used with respect for their public submission process.

---

## License

This documentation and exploration scripts are part of the UI-Agent repository and follow the repository's license. The form content documented here is owned by Anthropic and is documented for educational purposes only.

---

**End of README**
