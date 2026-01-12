# Google Form Exploration: Connectors Directory - Server Review Form

**Date:** January 11, 2026
**Form URL:** https://docs.google.com/forms/d/e/1FAIpQLSeafJF2NDI7oYx1r8o0ycivCSVLNq92Mpc1FPxMKSw1CzDkqA/viewform
**Exploration Method:** Automated Selenium WebDriver + Manual HTML Analysis

---

## Form Overview

### Title
**Connectors Directory - Server Review Form**

### Description
Thank you for your interest in having your MCP server considered for Anthropic's Connectors Directory! This form helps us collect information about your MCP server to evaluate it for potential inclusion in our curated directory. We review all submissions to ensure they meet our standards and policies. If your MCP server is selected for inclusion, the information you provide here will be displayed on Anthropic's [Connectors Directory page](https://www.anthropic.com/connectors) and may also be displayed within Anthropic's products, including Claude.ai.

**Important Notes:**
- Review process typically takes 1-2 weeks
- Meeting requirements doesn't guarantee directory inclusion
- Anthropic reserves the right to approve/decline submissions based on quality, security, and strategic fit
- Form completion time: Approximately 15-20 minutes

---

## Form Structure

This form consists of a multi-page checklist structure with the following main sections:

1. **Introduction Page** - Form overview and requirements
2. **Pre-Submission Requirements Checklist** - Comprehensive checklist with 4 major categories

---

## Detailed Form Content

### Page 1: Introduction

The introduction page provides context about the Connectors Directory review process and what submitters can expect. It includes information about:

- Purpose of the form
- Expected review timeline
- Link to the Connectors Directory page
- Reminder that completion doesn't guarantee approval

**Call to Action:** "Next" button to proceed to the checklist

---

### Page 2: Pre-Submission Requirements Checklist

**Header:** "Before proceeding, confirm you have completed ALL of the following:"

This is the main page of the form, containing all requirements organized into 4 required sections. Each section contains multiple checkbox items that submitters must acknowledge.

---

## Section 1: Policy Compliance *

**Type:** Multiple Checkboxes (Required)
**Purpose:** Ensure MCP servers comply with Anthropic's directory policies

### Checklist Items:

1. ☐ **I have read and understand the MCP Directory Review Guidelines**

2. ☐ **My server complies with all 30+ policy requirements**

3. ☐ **My server does NOT enable cross-service automation**
   - Ensures servers don't automate interactions across multiple external services without explicit user authorization
   - Prevents potential security and privacy risks

4. ☐ **My server does NOT transfer money, cryptocurrency, or execute financial transactions**
   - Excludes financial transaction capabilities
   - Reduces liability and security concerns for directory inclusion

5. ☐ **My server is in "GA" (General Availability) or will be by the time it's published in our Directory**
   - Ensures only production-ready servers are listed
   - "GA" means stable, publicly available, and supported

---

## Section 2: Technical Requirements *

**Type:** Multiple Checkboxes (Required)
**Purpose:** Verify technical implementation meets security and functionality standards

### Checklist Items:

1. ☐ **OAuth 2.0 is fully implemented for ALL tools requiring authentication**
   - Industry-standard authentication protocol
   - Required for any tool that needs user credentials
   - Ensures secure authorization flows

2. ☐ **All tools include proper safety annotations (readOnlyHint, destructiveHint)**
   - `readOnlyHint`: Indicates tool only reads data, doesn't modify
   - `destructiveHint`: Warns when tool performs irreversible actions
   - Critical for user safety and informed consent

3. ☐ **Server is accessible via HTTPS (not HTTP)**
   - SSL/TLS encryption required
   - Protects data in transit
   - Industry best practice for security

4. ☐ **CORS is properly configured for browser-based authentication**
   - Cross-Origin Resource Sharing (CORS) headers must be set correctly
   - Enables secure browser-based OAuth flows
   - Prevents unauthorized cross-origin requests

5. ☐ **Claude.ai and Claude Code IP addresses are allowlisted (if applicable)**
   - Some servers use IP allowlisting for security
   - If used, must include Anthropic's service IPs
   - Ensures connectivity from Claude products

---

## Section 3: Documentation Requirements *

**Type:** Multiple Checkboxes (Required)
**Purpose:** Ensure users have access to comprehensive documentation

### Checklist Items:

1. ☐ **Complete server documentation is published and publicly accessible**
   - Documentation must be available online
   - No login required to view basic documentation
   - Helps users understand capabilities before connecting

2. ☐ **Documentation includes setup instructions, tool descriptions, and troubleshooting guide**
   - **Setup instructions:** How to install/configure the server
   - **Tool descriptions:** What each tool does, parameters, and use cases
   - **Troubleshooting:** Common issues and solutions

3. ☐ **Company privacy policy is published and accessible**
   - Clear statement of data handling practices
   - Required for GDPR/CCPA compliance
   - Builds user trust

4. ☐ **Terms of service are published and accessible**
   - Legal agreement between server provider and users
   - Defines acceptable use, limitations, and liabilities
   - Standard requirement for service providers

---

## Section 4: Testing Requirements *

**Type:** Multiple Checkboxes (Required)
**Purpose:** Enable Anthropic reviewers to test the server functionality

### Checklist Items:

1. ☐ **Test account with sample data is ready (if relevant)**
   - Provides Anthropic reviewers with working credentials
   - Should have representative sample data
   - Only required if server needs authentication

2. ☐ **Test credentials are valid for at least 30 days (if relevant)**
   - Ensures reviewers have sufficient time to complete evaluation
   - Prevents mid-review credential expiration
   - Only required if test account provided

3. ☐ **All server tools are functional and tested in the surfaces in which they'll be available (claude.ai, Claude Code, etc)**
   - Server must be tested end-to-end
   - Must work in actual Claude interfaces, not just in development
   - Confirms readiness for production listing

---

## Form Submission

After completing all checklist items, users can proceed to submit the form. The form validates that all required checkboxes are checked before allowing submission.

---

## Technical Details

### Form Type
Multi-page Google Form with progressive disclosure (page-by-page navigation)

### Question Types
- **Checkboxes:** All questions are checkbox-based (allow multiple selections)
- **Required Questions:** All 4 main sections are marked as required

### Navigation
- **"Next" button** to advance from intro to checklist
- No "Back" button visible (single-direction progression)
- Form maintains state if user navigates away and returns (if signed into Google)

### Total Questions
- **4 main sections** (Policy, Technical, Documentation, Testing)
- **17 total checkbox items** across all sections

### Estimated Completion Time
15-20 minutes (assuming documentation and credentials are already prepared)

---

## Insights for Agent Training

### Challenges for Automation

1. **Dynamic Form Loading:** Google Forms use dynamic JavaScript rendering, requiring wait times for elements to load

2. **Checkbox Detection:** Checkboxes use `role="checkbox"` ARIA attribute rather than standard HTML `<input type="checkbox">`

3. **Multi-page Navigation:** Form has multiple pages, but our initial script got stuck in a loop clicking "Next" on the same page repeatedly

4. **CSS Selectors:** Google Forms use obfuscated class names (e.g., `M7eMe`, `aDTYNe`) that may change over time

### Successful Extraction Strategies

1. **HTML Source Analysis:** Extracting `page_source.html` and parsing with regex/grep proved most reliable

2. **ARIA Labels:** Checkbox options are stored in `aria-label` attributes, which are stable identifiers

3. **Screenshot Capture:** Taking screenshots at each stage provides visual record for verification

4. **Scrolling + Viewport Capture:** For long forms, scrolling incrementally and capturing each section works better than multi-page navigation

### Recommendations for Future Agents

1. **Use HTML Source:** When available, parse HTML source rather than relying only on visual element detection

2. **ARIA Attributes:** Prioritize ARIA labels and roles over class names or IDs

3. **Idempotent Actions:** Ensure navigation actions (like clicking "Next") verify page state changed before repeating

4. **Timeout Detection:** Implement logic to detect when stuck in a loop (e.g., same content after 3+ iterations)

5. **Hybrid Approach:** Combine Selenium for navigation with HTML parsing for content extraction

---

## Repository Structure

```
UI-agent/
├── src/
│   └── bench/
│       ├── google_form_explorer.py      # Initial exploration script (had pagination issue)
│       └── manual_form_extraction.py    # Improved script with scrolling
├── logs/
│   └── form_explorations/
│       ├── page_1.png                   # Introduction page
│       ├── page_2.png                   # Start of checklist
│       ├── page_3.png - page_33.png     # Duplicate pages (script loop issue)
│       ├── full_form_top.png            # Top of form
│       ├── scroll_001.png - scroll_003.png  # Scrolling screenshots
│       ├── page_source.html             # Complete HTML source
│       ├── extracted_content.txt        # Raw extracted text
│       └── FORM_DOCUMENTATION.md        # This file
```

---

## Files Generated

| File | Size | Description |
|------|------|-------------|
| `page_source.html` | 144 KB | Complete HTML source of the form |
| `extracted_content.txt` | 177 KB | Raw text extraction (includes JS/CSS) |
| `FORM_DOCUMENTATION.md` | This file | Human-readable form documentation |
| `page_*.png` | 33 files | Screenshots from initial exploration |
| `scroll_*.png` | 3 files | Screenshots from improved scrolling approach |

---

## Lessons Learned

### What Worked Well
- Selenium Firefox driver installed successfully via downloaded geckodriver
- HTML source extraction provided complete form structure
- ARIA labels gave accurate checkbox text
- Screenshots provided visual confirmation of form progression

### What Didn't Work
- Initial pagination detection (kept clicking same element)
- Direct text extraction from JavaScript-rendered content (too much noise)
- Relying on CSS classes for element selection (obfuscated names)

### Future Improvements
- Implement vision model (GPT-4V/Claude with vision) to read screenshots directly
- Add form state hash comparison to detect pagination loops
- Create reusable Google Forms parser class
- Integrate with Anthropic's Computer Use API for more natural interaction

---

## Example Use Cases for Training

This exploration provides training data for:

1. **Form Navigation Agents:** Learning to detect and avoid pagination loops
2. **Content Extraction Agents:** Parsing complex JavaScript-rendered forms
3. **Checklist Completion Agents:** Understanding structured requirement validation
4. **Documentation Agents:** Extracting and organizing form content for human readability

---

## Appendix: Full Checklist (Quick Reference)

### Policy Compliance (5 items)
- ☐ Read and understand MCP Directory Review Guidelines
- ☐ Complies with all 30+ policy requirements
- ☐ Does NOT enable cross-service automation
- ☐ Does NOT transfer money/cryptocurrency
- ☐ Server is in "GA" status

### Technical Requirements (5 items)
- ☐ OAuth 2.0 fully implemented
- ☐ Safety annotations on all tools
- ☐ HTTPS access
- ☐ CORS properly configured
- ☐ Anthropic IPs allowlisted

### Documentation Requirements (4 items)
- ☐ Complete documentation published
- ☐ Includes setup, tool descriptions, troubleshooting
- ☐ Privacy policy published
- ☐ Terms of service published

### Testing Requirements (3 items)
- ☐ Test account ready (if relevant)
- ☐ Credentials valid for 30+ days
- ☐ Tools tested in Claude.ai and Claude Code

**Total: 17 checklist items across 4 required sections**

---

**End of Documentation**
