"""
Manual Form Extraction Script
Scrolls through the form and captures content systematically
"""

import os
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException


class ManualFormExtractor:
    def __init__(self, form_url: str, output_dir: str = "logs/form_explorations"):
        self.form_url = form_url
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Setup Firefox driver
        options = Options()
        self.driver = webdriver.Firefox(options=options)
        self.driver.set_window_size(1920, 1080)
        self.wait = WebDriverWait(self.driver, 10)

        self.extracted_content = []

    def scroll_and_extract(self):
        """Scroll through the form and extract all visible text"""
        print(f"Opening form: {self.form_url}")
        self.driver.get(self.form_url)
        time.sleep(3)

        # Click "Next" to get past intro page
        try:
            next_button = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, "//span[text()='Next' or text()='Next' or text()='Continue']"))
            )
            next_button.click()
            time.sleep(2)
        except:
            print("No initial 'Next' button found, continuing...")

        # Take initial screenshot
        self.driver.save_screenshot(os.path.join(self.output_dir, "full_form_top.png"))

        # Get the entire page body text
        try:
            body = self.driver.find_element(By.TAG_NAME, "body")

            # Scroll down incrementally and capture screenshots
            scroll_position = 0
            scroll_increment = 800
            screenshot_num = 1

            while True:
                # Scroll down
                self.driver.execute_script(f"window.scrollTo(0, {scroll_position});")
                time.sleep(0.5)

                # Take screenshot
                screenshot_path = os.path.join(self.output_dir, f"scroll_{screenshot_num:03d}.png")
                self.driver.save_screenshot(screenshot_path)
                print(f"Screenshot {screenshot_num}: scroll position {scroll_position}")

                # Try to get all text elements visible
                visible_text = self.driver.execute_script("""
                    var elements = document.querySelectorAll('*');
                    var text = [];
                    for (var i = 0; i < elements.length; i++) {
                        var elem = elements[i];
                        var style = window.getComputedStyle(elem);
                        if (style.display !== 'none' &&
                            style.visibility !== 'hidden' &&
                            elem.offsetHeight > 0 &&
                            elem.textContent.trim() !== '') {
                            var rect = elem.getBoundingClientRect();
                            if (rect.top >= 0 && rect.top < window.innerHeight) {
                                text.push(elem.textContent.trim());
                            }
                        }
                    }
                    return text.join('\\n---\\n');
                """)

                if visible_text:
                    self.extracted_content.append(f"\n=== Scroll Position {scroll_position} ===\n{visible_text}\n")

                # Check if we've reached the bottom
                scroll_height = self.driver.execute_script("return document.body.scrollHeight")
                viewport_height = self.driver.execute_script("return window.innerHeight")

                if scroll_position + viewport_height >= scroll_height:
                    print("Reached bottom of page")
                    break

                scroll_position += scroll_increment
                screenshot_num += 1

                if screenshot_num > 50:  # Safety limit
                    print("Reached maximum scroll iterations")
                    break

        except Exception as e:
            print(f"Error during scrolling: {e}")
            import traceback
            traceback.print_exc()

        # Get the full page source as fallback
        page_source = self.driver.page_source
        source_path = os.path.join(self.output_dir, "page_source.html")
        with open(source_path, 'w', encoding='utf-8') as f:
            f.write(page_source)
        print(f"\nPage source saved to: {source_path}")

        # Save extracted content
        content_path = os.path.join(self.output_dir, "extracted_content.txt")
        with open(content_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(self.extracted_content))
        print(f"Extracted content saved to: {content_path}")

        # Keep browser open for manual inspection
        print("\n" + "="*50)
        print("Browser will remain open for manual inspection.")
        print("Press Ctrl+C to close.")
        print("="*50)

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nClosing browser...")

    def cleanup(self):
        self.driver.quit()


def main():
    form_url = "https://docs.google.com/forms/d/e/1FAIpQLSeafJF2NDI7oYx1r8o0ycivCSVLNq92Mpc1FPxMKSw1CzDkqA/viewform"

    extractor = ManualFormExtractor(form_url)

    try:
        extractor.scroll_and_extract()
    except KeyboardInterrupt:
        print("\n\nExtraction interrupted by user")
    except Exception as e:
        print(f"\n\nError during extraction: {e}")
        import traceback
        traceback.print_exc()
    finally:
        extractor.cleanup()


if __name__ == "__main__":
    main()
