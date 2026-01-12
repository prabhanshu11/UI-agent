"""
Google Form Explorer
Navigates through a Google Form and documents all questions, options, and pages.
"""

import os
import json
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException


class GoogleFormExplorer:
    def __init__(self, form_url: str, output_dir: str = "logs/form_explorations"):
        self.form_url = form_url
        self.output_dir = output_dir
        self.form_data = {
            "url": form_url,
            "timestamp": datetime.now().isoformat(),
            "pages": [],
            "metadata": {}
        }

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Setup Firefox driver
        options = Options()
        # Run in headless mode (set to False to see the browser)
        # options.add_argument('--headless')
        self.driver = webdriver.Firefox(options=options)
        self.driver.set_window_size(1920, 1080)
        self.wait = WebDriverWait(self.driver, 10)

    def extract_form_title(self):
        """Extract the form title and description"""
        try:
            title_elem = self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "[role='heading']"))
            )
            self.form_data["metadata"]["title"] = title_elem.text

            # Try to get description
            try:
                desc_elem = self.driver.find_element(By.CSS_SELECTOR, ".freebirdFormviewerViewHeaderDescription")
                self.form_data["metadata"]["description"] = desc_elem.text
            except NoSuchElementException:
                self.form_data["metadata"]["description"] = ""

        except TimeoutException:
            print("Could not find form title")

    def extract_questions_from_page(self, page_number: int):
        """Extract all questions from the current page"""
        print(f"\n=== Extracting Page {page_number} ===")

        page_data = {
            "page_number": page_number,
            "questions": []
        }

        # Wait for questions to load
        time.sleep(1)

        # Find all question containers
        # Google Forms uses different class names, we'll try multiple selectors
        question_selectors = [
            ".freebirdFormviewerComponentsQuestionBaseRoot",
            "[role='listitem']",
            ".Qr7Oae"  # Another common class for question containers
        ]

        questions = []
        for selector in question_selectors:
            try:
                questions = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if questions:
                    break
            except:
                continue

        if not questions:
            print("No questions found on this page")
            return page_data

        print(f"Found {len(questions)} question elements")

        for idx, question_elem in enumerate(questions, 1):
            try:
                question_data = self.extract_question(question_elem, idx)
                if question_data:
                    page_data["questions"].append(question_data)
            except Exception as e:
                print(f"Error extracting question {idx}: {e}")
                continue

        return page_data

    def extract_question(self, question_elem, question_num: int):
        """Extract details of a single question"""
        question_data = {
            "number": question_num,
            "text": "",
            "type": "",
            "required": False,
            "options": []
        }

        try:
            # Extract question text
            # Try multiple selectors for question text
            text_selectors = [
                ".freebirdFormviewerComponentsQuestionBaseTitle",
                "[role='heading']",
                ".M7eMe"
            ]

            for selector in text_selectors:
                try:
                    text_elem = question_elem.find_element(By.CSS_SELECTOR, selector)
                    question_data["text"] = text_elem.text.strip()
                    if question_data["text"]:
                        break
                except:
                    continue

            # Check if required
            try:
                required_elem = question_elem.find_element(By.CSS_SELECTOR, "[aria-label*='Required']")
                question_data["required"] = True
            except NoSuchElementException:
                question_data["required"] = False

            # Extract question type and options
            # Multiple choice radio buttons
            if self.has_radio_buttons(question_elem):
                question_data["type"] = "multiple_choice"
                question_data["options"] = self.extract_radio_options(question_elem)

            # Checkboxes
            elif self.has_checkboxes(question_elem):
                question_data["type"] = "checkboxes"
                question_data["options"] = self.extract_checkbox_options(question_elem)

            # Dropdown
            elif self.has_dropdown(question_elem):
                question_data["type"] = "dropdown"
                question_data["options"] = self.extract_dropdown_options(question_elem)

            # Text input (short answer)
            elif self.has_text_input(question_elem):
                question_data["type"] = "short_answer"

            # Paragraph text
            elif self.has_paragraph_input(question_elem):
                question_data["type"] = "paragraph"

            # Date/Time
            elif self.has_date_input(question_elem):
                question_data["type"] = "date"

            # Linear scale
            elif self.has_linear_scale(question_elem):
                question_data["type"] = "linear_scale"
                question_data["options"] = self.extract_linear_scale_options(question_elem)

            else:
                question_data["type"] = "unknown"

            print(f"Q{question_num}: {question_data['text'][:50]}... ({question_data['type']})")

        except Exception as e:
            print(f"Error in extract_question: {e}")

        return question_data

    def has_radio_buttons(self, elem):
        try:
            elem.find_element(By.CSS_SELECTOR, "[role='radio']")
            return True
        except:
            return False

    def extract_radio_options(self, elem):
        options = []
        try:
            radio_elements = elem.find_elements(By.CSS_SELECTOR, "[role='radio']")
            for radio in radio_elements:
                # Get the label text
                try:
                    label = radio.find_element(By.XPATH, ".//ancestor::*[contains(@class, 'freebirdFormviewerComponentsQuestionRadioChoice') or contains(@class, 'Od2TWd')]")
                    option_text = label.text.strip()
                    if option_text:
                        options.append(option_text)
                except:
                    pass
        except Exception as e:
            print(f"Error extracting radio options: {e}")
        return options

    def has_checkboxes(self, elem):
        try:
            elem.find_element(By.CSS_SELECTOR, "[role='checkbox']")
            return True
        except:
            return False

    def extract_checkbox_options(self, elem):
        options = []
        try:
            checkbox_elements = elem.find_elements(By.CSS_SELECTOR, "[role='checkbox']")
            for checkbox in checkbox_elements:
                try:
                    label = checkbox.find_element(By.XPATH, ".//ancestor::*[contains(@class, 'freebirdFormviewerComponentsQuestionCheckboxChoice')]")
                    option_text = label.text.strip()
                    if option_text:
                        options.append(option_text)
                except:
                    pass
        except:
            pass
        return options

    def has_dropdown(self, elem):
        try:
            elem.find_element(By.CSS_SELECTOR, "[role='listbox'], select")
            return True
        except:
            return False

    def extract_dropdown_options(self, elem):
        options = []
        try:
            select_elem = elem.find_element(By.CSS_SELECTOR, "[role='listbox'], select")
            # Click to open dropdown
            select_elem.click()
            time.sleep(0.5)

            option_elements = self.driver.find_elements(By.CSS_SELECTOR, "[role='option']")
            for option in option_elements:
                option_text = option.text.strip()
                if option_text:
                    options.append(option_text)

            # Click again to close
            select_elem.click()
        except:
            pass
        return options

    def has_text_input(self, elem):
        try:
            elem.find_element(By.CSS_SELECTOR, "input[type='text']:not([role='listbox'])")
            return True
        except:
            return False

    def has_paragraph_input(self, elem):
        try:
            elem.find_element(By.CSS_SELECTOR, "textarea")
            return True
        except:
            return False

    def has_date_input(self, elem):
        try:
            elem.find_element(By.CSS_SELECTOR, "input[type='date']")
            return True
        except:
            return False

    def has_linear_scale(self, elem):
        try:
            elem.find_element(By.CSS_SELECTOR, "[role='radiogroup']")
            return True
        except:
            return False

    def extract_linear_scale_options(self, elem):
        options = []
        try:
            scale_elements = elem.find_elements(By.CSS_SELECTOR, "[role='radio']")
            for scale in scale_elements:
                label_text = scale.get_attribute("aria-label")
                if label_text:
                    options.append(label_text)
        except:
            pass
        return options

    def has_next_button(self):
        """Check if there's a 'Next' button on the current page"""
        try:
            next_buttons = self.driver.find_elements(By.CSS_SELECTOR, "[role='button']")
            for btn in next_buttons:
                if btn.text.lower() in ['next', 'continue', 'नेक्स्ट', 'आगे']:
                    return btn
            return None
        except:
            return None

    def explore_form(self):
        """Main method to explore the entire form"""
        print(f"Opening form: {self.form_url}")
        self.driver.get(self.form_url)

        # Extract form title
        self.extract_form_title()
        print(f"\nForm Title: {self.form_data['metadata'].get('title', 'N/A')}")

        page_number = 1

        while True:
            # Extract questions from current page
            page_data = self.extract_questions_from_page(page_number)
            self.form_data["pages"].append(page_data)

            # Take screenshot of the page
            screenshot_path = os.path.join(self.output_dir, f"page_{page_number}.png")
            self.driver.save_screenshot(screenshot_path)
            print(f"Screenshot saved: {screenshot_path}")

            # Check for next button
            next_button = self.has_next_button()
            if next_button:
                print(f"\nNavigating to next page...")
                next_button.click()
                time.sleep(2)  # Wait for page to load
                page_number += 1
            else:
                print("\nNo more pages found. Exploration complete!")
                break

        # Save the data
        self.save_data()

        print(f"\n{'='*50}")
        print(f"Exploration Summary:")
        print(f"Total Pages: {len(self.form_data['pages'])}")
        total_questions = sum(len(page['questions']) for page in self.form_data['pages'])
        print(f"Total Questions: {total_questions}")
        print(f"{'='*50}")

    def save_data(self):
        """Save the extracted form data to JSON and text files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON
        json_path = os.path.join(self.output_dir, f"form_data_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.form_data, f, indent=2, ensure_ascii=False)
        print(f"\nJSON data saved: {json_path}")

        # Save human-readable text
        text_path = os.path.join(self.output_dir, f"form_content_{timestamp}.txt")
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(f"Google Form Exploration Report\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"URL: {self.form_data['url']}\n")
            f.write(f"Timestamp: {self.form_data['timestamp']}\n")
            f.write(f"Title: {self.form_data['metadata'].get('title', 'N/A')}\n")
            f.write(f"Description: {self.form_data['metadata'].get('description', 'N/A')}\n")
            f.write(f"\n{'='*80}\n\n")

            for page in self.form_data['pages']:
                f.write(f"\nPAGE {page['page_number']}\n")
                f.write(f"{'-'*80}\n")

                for q in page['questions']:
                    f.write(f"\nQ{q['number']}. {q['text']}")
                    if q['required']:
                        f.write(" *")
                    f.write(f"\n   Type: {q['type']}\n")

                    if q['options']:
                        f.write("   Options:\n")
                        for opt in q['options']:
                            f.write(f"   - {opt}\n")
                    f.write("\n")

        print(f"Text report saved: {text_path}")

    def cleanup(self):
        """Close the browser"""
        self.driver.quit()


def main():
    form_url = "https://docs.google.com/forms/d/e/1FAIpQLSeafJF2NDI7oYx1r8o0ycivCSVLNq92Mpc1FPxMKSw1CzDkqA/viewform"

    explorer = GoogleFormExplorer(form_url)

    try:
        explorer.explore_form()
    except KeyboardInterrupt:
        print("\n\nExploration interrupted by user")
    except Exception as e:
        print(f"\n\nError during exploration: {e}")
        import traceback
        traceback.print_exc()
    finally:
        explorer.cleanup()


if __name__ == "__main__":
    main()
