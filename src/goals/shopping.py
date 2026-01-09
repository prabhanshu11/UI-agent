"""Shopping-specific goal definitions."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ShoppingGoal:
    """Base class for shopping goals."""
    platform: str  # amazon, swiggy, blinkit, ubereats
    intent: str = ""


@dataclass
class NavigateToProduct(ShoppingGoal):
    """Goal to navigate to a product page."""
    intent: str = "Navigate to product"
    product_id: str = ""
    product_url: str = ""


@dataclass
class AddToCart(ShoppingGoal):
    """Goal to add a product to cart."""
    intent: str = "Add to cart"
    product_id: str = ""
    quantity: int = 1


@dataclass
class ViewCart(ShoppingGoal):
    """Goal to view the shopping cart."""
    intent: str = "View cart"


@dataclass
class VerifyAddress(ShoppingGoal):
    """Goal to verify delivery address for cart items."""
    intent: str = "Verify delivery address"
    expected_address: str = ""


@dataclass
class SearchProducts(ShoppingGoal):
    """Goal to search for products."""
    intent: str = "Search products"
    query: str = ""
    limit: int = 10
