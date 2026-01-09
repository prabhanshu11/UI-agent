"""FastAPI server for UI-Agent browser automation."""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.browser.controller import BrowserController, BrowserConfig

logger = logging.getLogger(__name__)

# Global browser controller instance
_browser: BrowserController | None = None


async def get_browser() -> BrowserController:
    """Get the browser controller, starting it if needed."""
    global _browser
    if _browser is None:
        _browser = BrowserController(BrowserConfig(headless=True))
        await _browser.start()
    return _browser


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    yield
    global _browser
    if _browser:
        await _browser.stop()
        _browser = None


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    return FastAPI(
        title="UI-Agent API",
        description="Browser automation API for shopping agent integration",
        version="0.2.0",
        lifespan=lifespan,
    )


app = create_app()


class NavigateRequest(BaseModel):
    """Request to navigate to a URL."""
    url: str
    wait_until: str = "domcontentloaded"


class ClickRequest(BaseModel):
    """Request to click an element."""
    selector: str


class FillRequest(BaseModel):
    """Request to fill a form field."""
    selector: str
    value: str


class TypeRequest(BaseModel):
    """Request to type text."""
    selector: str
    text: str
    delay: int = 50


class SelectRequest(BaseModel):
    """Request to select an option."""
    selector: str
    value: str


class WaitRequest(BaseModel):
    """Request to wait for element."""
    selector: str
    timeout: int | None = None


class EvaluateRequest(BaseModel):
    """Request to evaluate JavaScript."""
    script: str


class AmazonAddToCartRequest(BaseModel):
    """Request to add Amazon product to cart."""
    product_id: str
    quantity: int = 1


class AmazonCartRequest(BaseModel):
    """Request to get Amazon cart contents."""
    pass


class AmazonVerifyAddressRequest(BaseModel):
    """Request to verify Amazon delivery address."""
    pass


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.2.0"}


@app.post("/browser/start")
async def start_browser(headless: bool = True):
    """Start the browser."""
    global _browser
    if _browser is not None:
        return {"status": "already_running"}

    _browser = BrowserController(BrowserConfig(headless=headless))
    await _browser.start()
    return {"status": "started"}


@app.post("/browser/stop")
async def stop_browser():
    """Stop the browser."""
    global _browser
    if _browser is None:
        return {"status": "not_running"}

    await _browser.stop()
    _browser = None
    return {"status": "stopped"}


@app.post("/browser/navigate")
async def navigate(request: NavigateRequest):
    """Navigate to a URL."""
    browser = await get_browser()
    result = await browser.navigate(request.url, request.wait_until)
    return result


@app.post("/browser/click")
async def click(request: ClickRequest):
    """Click an element."""
    browser = await get_browser()
    try:
        result = await browser.click(request.selector)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/browser/fill")
async def fill(request: FillRequest):
    """Fill a form field."""
    browser = await get_browser()
    try:
        result = await browser.fill(request.selector, request.value)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/browser/type")
async def type_text(request: TypeRequest):
    """Type text into an element."""
    browser = await get_browser()
    try:
        result = await browser.type_text(request.selector, request.text, request.delay)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/browser/select")
async def select_option(request: SelectRequest):
    """Select an option from dropdown."""
    browser = await get_browser()
    try:
        result = await browser.select_option(request.selector, request.value)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/browser/wait")
async def wait_for_selector(request: WaitRequest):
    """Wait for an element to appear."""
    browser = await get_browser()
    found = await browser.wait_for_selector(request.selector, request.timeout)
    return {"found": found, "selector": request.selector}


@app.get("/browser/screenshot")
async def screenshot(full_page: bool = False):
    """Take a screenshot."""
    browser = await get_browser()
    screenshot_bytes = await browser.screenshot(full_page=full_page)
    from fastapi.responses import Response
    return Response(content=screenshot_bytes, media_type="image/png")


@app.get("/browser/content")
async def get_content():
    """Get page content."""
    browser = await get_browser()
    return await browser.get_page_content()


@app.post("/browser/evaluate")
async def evaluate(request: EvaluateRequest):
    """Evaluate JavaScript."""
    browser = await get_browser()
    try:
        result = await browser.evaluate(request.script)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/amazon/add-to-cart")
async def amazon_add_to_cart(request: AmazonAddToCartRequest):
    """
    Add a product to Amazon cart.

    This navigates to the product page, sets quantity, and clicks Add to Cart.
    """
    browser = await get_browser()

    try:
        product_url = f"https://www.amazon.in/dp/{request.product_id}"
        await browser.navigate(product_url, wait_until="networkidle")

        if request.quantity > 1:
            quantity_found = await browser.wait_for_selector("#quantity", timeout=5000)
            if quantity_found:
                await browser.select_option("#quantity", str(request.quantity))

        add_to_cart_selector = "#add-to-cart-button"
        found = await browser.wait_for_selector(add_to_cart_selector, timeout=10000)

        if not found:
            return {
                "success": False,
                "message": "Add to Cart button not found",
                "product_id": request.product_id,
            }

        await browser.click(add_to_cart_selector)

        await browser.wait_for_navigation(timeout=10000)

        return {
            "success": True,
            "message": f"Added {request.quantity}x {request.product_id} to cart",
            "product_id": request.product_id,
            "quantity": request.quantity,
        }

    except Exception as e:
        logger.error("Failed to add to cart: %s", str(e))
        return {
            "success": False,
            "message": str(e),
            "product_id": request.product_id,
        }


@app.get("/amazon/cart")
async def amazon_get_cart():
    """
    Get Amazon cart contents.

    Navigates to cart page and extracts items.
    """
    browser = await get_browser()

    try:
        await browser.navigate("https://www.amazon.in/gp/cart/view.html", wait_until="networkidle")

        items_script = """
        () => {
            const items = [];
            const cartItems = document.querySelectorAll('[data-asin]');
            cartItems.forEach(item => {
                const asin = item.getAttribute('data-asin');
                if (asin) {
                    const titleEl = item.querySelector('.sc-product-title');
                    const priceEl = item.querySelector('.sc-product-price');
                    const quantityEl = item.querySelector('.sc-quantity-textfield input');
                    items.push({
                        asin: asin,
                        title: titleEl ? titleEl.textContent.trim() : '',
                        price: priceEl ? priceEl.textContent.trim() : '',
                        quantity: quantityEl ? parseInt(quantityEl.value) : 1,
                    });
                }
            });
            return items;
        }
        """

        items = await browser.evaluate(items_script)

        subtotal_script = """
        () => {
            const el = document.querySelector('#sc-subtotal-amount-activecart');
            return el ? el.textContent.trim() : null;
        }
        """
        subtotal = await browser.evaluate(subtotal_script)

        return {
            "items": items,
            "subtotal": subtotal,
            "item_count": len(items),
        }

    except Exception as e:
        logger.error("Failed to get cart: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/amazon/verify-address")
async def amazon_verify_address():
    """
    Verify delivery address for Amazon cart.

    Checks if all items can be delivered to the selected address.
    """
    browser = await get_browser()

    try:
        await browser.navigate("https://www.amazon.in/gp/cart/view.html", wait_until="networkidle")

        address_script = """
        () => {
            const addressEl = document.querySelector('#contextualIngressPtLabel_deliveryShortLine');
            return addressEl ? addressEl.textContent.trim() : null;
        }
        """
        current_address = await browser.evaluate(address_script)

        undeliverable_script = """
        () => {
            const undeliverable = document.querySelectorAll('.sc-undeliverable-item');
            return undeliverable.length;
        }
        """
        undeliverable_count = await browser.evaluate(undeliverable_script)

        return {
            "valid": undeliverable_count == 0,
            "current_address": current_address,
            "undeliverable_items": undeliverable_count,
            "message": "All items deliverable" if undeliverable_count == 0 else f"{undeliverable_count} items cannot be delivered",
        }

    except Exception as e:
        logger.error("Failed to verify address: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/amazon/search")
async def amazon_search(query: str, limit: int = 10):
    """
    Search for products on Amazon.

    Args:
        query: Search query
        limit: Maximum results to return
    """
    browser = await get_browser()

    try:
        search_url = f"https://www.amazon.in/s?k={query.replace(' ', '+')}"
        await browser.navigate(search_url, wait_until="networkidle")

        products_script = f"""
        () => {{
            const products = [];
            const items = document.querySelectorAll('[data-asin]:not([data-asin=""])');
            let count = 0;
            items.forEach(item => {{
                if (count >= {limit}) return;
                const asin = item.getAttribute('data-asin');
                const titleEl = item.querySelector('h2 a span');
                const priceEl = item.querySelector('.a-price .a-offscreen');
                const ratingEl = item.querySelector('.a-icon-star-small .a-icon-alt');
                const imageEl = item.querySelector('.s-image');

                if (titleEl) {{
                    products.push({{
                        product_id: asin,
                        name: titleEl.textContent.trim(),
                        price: priceEl ? parseFloat(priceEl.textContent.replace(/[^0-9.]/g, '')) : null,
                        currency: 'INR',
                        rating: ratingEl ? parseFloat(ratingEl.textContent.split(' ')[0]) : null,
                        image_url: imageEl ? imageEl.src : null,
                        url: 'https://www.amazon.in/dp/' + asin,
                    }});
                    count++;
                }}
            }});
            return products;
        }}
        """

        products = await browser.evaluate(products_script)

        return {
            "query": query,
            "results": products,
            "count": len(products),
        }

    except Exception as e:
        logger.error("Failed to search: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
