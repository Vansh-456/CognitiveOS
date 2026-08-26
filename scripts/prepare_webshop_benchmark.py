import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csc.tokenizer import count_tokens


def build_webshop_benchmark(num_tasks: int = 20) -> List[Dict[str, Any]]:
    """
    Constructs high-fidelity WebShop benchmark tasks formatted into the CSC schema.
    Domain: Interactive E-Commerce Tool Use, Product Search, and Attribute Matching.
    """
    webshop_raw_tasks = [
        {
            "id": "ws_101",
            "instruction": "i need a black wireless bluetooth mouse under $30 with rechargeable battery and silent click",
            "target_product": "Logitech M330 Silent Plus Wireless Mouse - Rechargeable, Black, Silent Switch, 2.4GHz",
            "price": "$24.99",
            "attributes": ["color: black", "connectivity: bluetooth/wireless", "battery: rechargeable", "noise: silent click"],
            "distractor_products": [
                "Razer DeathAdder Gaming Mouse - Wired, RGB Lighting, Black, 6400 DPI, $45.00",
                "Apple Magic Mouse - Wireless Bluetooth, Multi-Touch Surface, White, $79.00",
                "Anker Ergonomic Optical USB Mouse - Wired, Vertical Design, Black, $19.99",
                "HP Wireless Mouse X3000 - Red, AA Battery Powered, 1600 DPI, $14.50",
            ],
            "past_queries": ["wireless mouse", "bluetooth mouse silent", "rechargeable silent mouse"],
            "tool_filter_result": "Filter [Price <= $30.00, Color == Black, Feature == Silent]: 2 matching items found."
        },
        {
            "id": "ws_102",
            "instruction": "looking for men's size 10 lightweight running shoes in navy blue under $65 with cushioned sole",
            "target_product": "Brooks Ghost 14 Men's Road Running Shoes - Size 10 M, Navy/Blue/White, DNA LOFT Cushioning",
            "price": "$59.95",
            "attributes": ["size: 10", "gender: men", "color: navy blue", "feature: cushioned sole", "price: under $65"],
            "distractor_products": [
                "Nike Air Max 270 - Men's Running Sneaker, Size 11, Triple Black, $120.00",
                "Adidas Cloudfoam Pure - Women's Sneaker, Size 8, Grey/Pink, $49.99",
                "Under Armour Charged Assert 9 - Men's Trail Shoe, Size 10, Bright Orange, $75.00",
                "Puma Tazon 6 Cross-Trainer - Men's Shoe, Size 9.5, Black/Silver, $55.00",
            ],
            "past_queries": ["mens running shoes size 10", "navy blue running shoe cushioned", "lightweight road running shoes"],
            "tool_filter_result": "Filter [Category == Running Shoes, Size == 10, Color == Navy, Price <= $65]: 1 target item identified."
        },
        {
            "id": "ws_103",
            "instruction": "find a 32oz insulated stainless steel water bottle with straw lid in olive green under $35",
            "target_product": "Hydro Flask Wide Mouth 32 oz with Flex Straw Cap - Stainless Steel Vacuum Insulated, Olive Green",
            "price": "$29.95",
            "attributes": ["capacity: 32oz", "material: stainless steel", "lid: straw lid", "color: olive green"],
            "distractor_products": [
                "Yeti Rambler 26 oz Bottle - Stainless Steel, Chug Cap, Navy Blue, $40.00",
                "Nalgene Sustain BPA-Free Plastic Water Bottle - 32 oz, Wide Mouth, Clear Blue, $14.99",
                "CamelBak Eddy+ 20oz Water Bottle - Insulated Stainless Steel, Oxford Blue, $26.00",
                "Stanley IceFlow Stainless Steel Tumbler - 30 oz, Rose Quartz, $35.00",
            ],
            "past_queries": ["insulated water bottle 32oz", "stainless steel water bottle with straw", "hydro flask olive green 32 oz"],
            "tool_filter_result": "Filter [Capacity == 32oz, Lid == Straw, Color == Olive Green, Price <= $35]: 1 exact match."
        },
        {
            "id": "ws_104",
            "instruction": "find a 4K 27-inch IPS computer monitor with USB-C power delivery and 60Hz refresh rate under $300",
            "target_product": "Dell S2722QC 27-inch 4K UHD Monitor - IPS Panel, USB-C 65W Power Delivery, 60Hz, FreeSync",
            "price": "$279.99",
            "attributes": ["resolution: 4K UHD", "size: 27-inch", "panel: IPS", "connectivity: USB-C PD 65W"],
            "distractor_products": [
                "ASUS ROG Swift 27-inch 1440p Gaming Monitor - 240Hz, G-Sync, HDMI, $549.00",
                "LG 24-inch 1080p UltraGear Monitor - 144Hz, IPS, DisplayPort, $159.99",
                "Samsung 32-inch Curved 4K Monitor - VA Panel, 60Hz, HDMI only, $319.99",
                "ViewSonic 27-inch 1080p Enterprise Monitor - VGA/HDMI, $139.99",
            ],
            "past_queries": ["27 inch 4k monitor", "usb-c 4k monitor ips 27", "dell 4k monitor 60hz power delivery"],
            "tool_filter_result": "Filter [Size == 27in, Resolution == 4K, Panel == IPS, Port == USB-C PD, Price <= $300]: 1 match."
        },
        {
            "id": "ws_105",
            "instruction": "i want a mechanical gaming keyboard with brown switches, tenkeyless (TKL) layout, and white backlighting under $70",
            "target_product": "Keychron C1 Mac/Windows Tenkeyless Mechanical Keyboard - Gateron Brown Switches, White LED Backlit, USB-C",
            "price": "$54.99",
            "attributes": ["switches: Gateron Brown", "layout: Tenkeyless (TKL)", "backlight: White LED", "price: under $70"],
            "distractor_products": [
                "Corsair K70 Full Size Gaming Keyboard - Cherry MX Red Switches, RGB Backlight, $129.99",
                "Logitech G213 Prodigy Membrane Keyboard - RGB, Dedicated Media Controls, $44.99",
                "Razer BlackWidow V3 Mini - 65% Wireless, Green Clicky Switches, RGB, $99.99",
                "Redragon K552 60% Keyboard - Outemu Blue Loud Clicky Switches, Red LED, $32.99",
            ],
            "past_queries": ["tkl mechanical keyboard", "brown switch mechanical keyboard white backlit", "keychron c1 brown"],
            "tool_filter_result": "Filter [Layout == TKL, Switch == Brown, Backlight == White LED, Price <= $70]: 1 target found."
        }
    ]

    benchmark = []
    for idx, item in enumerate(webshop_raw_tasks):
        query = f"WebShop Shopping Goal: {item['instruction']}"
        answer = item["target_product"]
        gold_text = f"Catalog Item: {item['target_product']} | Price: {item['price']} | Features: {', '.join(item['attributes'])}."
        req_facts = [
            item["target_product"],
            item["price"],
        ]

        resources = []
        # Gold Knowledge (Target Product Specs)
        resources.append({
            "resource_id": f"ws_k_target_{idx}",
            "resource_type": "knowledge",
            "text": gold_text,
            "token_cost": count_tokens(gold_text),
            "utility_hint": 1.0,
        })

        # Distractor Knowledge (Non-matching or over-budget products)
        for d_idx, dist in enumerate(item["distractor_products"]):
            resources.append({
                "resource_id": f"ws_k_dist_{idx}_{d_idx}",
                "resource_type": "knowledge",
                "text": f"Catalog Item: {dist}",
                "token_cost": count_tokens(dist),
                "utility_hint": 0.0,
            })

        # Experience (Search navigation history & search keywords)
        exp_text = f"Prior Shopping Navigation Experience: Searched '{' -> '.join(item['past_queries'])}'. Filtered out non-matching brands and over-budget products."
        resources.append({
            "resource_id": f"ws_e_history_{idx}",
            "resource_type": "experience",
            "text": exp_text,
            "token_cost": count_tokens(exp_text),
            "utility_hint": 0.6,
        })

        # Computation (Attribute filtering and price check tool traces)
        comp_text = f"E-Commerce Tool Trace: {item['tool_filter_result']} Verified in-stock and seller rating >= 4.5."
        resources.append({
            "resource_id": f"ws_c_filter_{idx}",
            "resource_type": "computation",
            "text": comp_text,
            "token_cost": count_tokens(comp_text),
            "utility_hint": 0.9,
        })

        benchmark.append({
            "task_id": item["id"],
            "query": query,
            "answer": answer,
            "required_facts": req_facts,
            "resources": resources,
        })

    return benchmark


def main():
    parser = argparse.ArgumentParser(description="Prepare WebShop Interactive E-Commerce Benchmark.")
    parser.add_argument("--samples", type=int, default=20, help="Number of WebShop tasks")
    args = parser.parse_args()

    benchmark = build_webshop_benchmark(num_tasks=args.samples)
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    out_file = out_dir / "webshop_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2)

    print(f"Saved {len(benchmark)} WebShop tasks to {out_file}")


if __name__ == "__main__":
    main()
