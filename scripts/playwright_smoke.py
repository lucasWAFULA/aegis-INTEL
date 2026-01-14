from playwright.sync_api import sync_playwright
import time
import sys
import os

URL = os.environ.get("SMOKE_URL", "http://localhost:8504")

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()

    page.on("console", lambda msg: print("PAGE_CONSOLE", msg.type, msg.text))
    page.on("pageerror", lambda exc: print("PAGE_ERROR", exc))

    print("WAITING for server at", URL)
    import requests
    for attempt in range(30):
        try:
            r = requests.get(URL, timeout=3)
            print("HTTP", r.status_code)
            if r.status_code == 200:
                break
        except Exception as e:
            print("WAIT", attempt, e)
            time.sleep(1)
    else:
        print("SERVER_NOT_READY")
        sys.exit(2)

    print("NAVIGATING to", URL)
    page.goto(URL, wait_until="networkidle", timeout=30000)
    time.sleep(2)
    print("PAGE_TITLE", page.title())
    page.screenshot(path="d:/FINAL HUMINT DASH/playwright_smoke_initial.png", full_page=True)

    def click_text(t):
        try:
            btn = page.locator(f"button:has-text('{t}')").first
            btn.click(timeout=7000)
            print("CLICKED", t)
            time.sleep(1.5)
            page.screenshot(path=f"d:/FINAL HUMINT DASH/playwright_smoke_{t.replace(' ','_')}.png", full_page=True)
        except Exception as e:
            print("NO_BUTTON", t, str(e))

    candidates = ["Run Optimisation", "Run Optimization", "Run optimisation", "Explain Decision", "Explain Source", "Explain"]
    for c in candidates:
        click_text(c)

    # capture main content snippet
    content = page.content()[:4000]
    print("HTML_SNIPPET_START\n", content)

    # wait a bit then close
    time.sleep(1)
    browser.close()
    print("DONE")
